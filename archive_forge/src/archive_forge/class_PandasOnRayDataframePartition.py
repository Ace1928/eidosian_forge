from typing import TYPE_CHECKING, Callable, Union
import pandas
import ray
from modin.config import LazyExecution
from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from modin.core.execution.ray.common import MaterializationHook, RayWrapper
from modin.core.execution.ray.common.deferred_execution import (
from modin.core.execution.ray.common.utils import ObjectIDType
from modin.logging import disable_logging, get_logger
from modin.pandas.indexing import compute_sliced_len
from modin.utils import _inherit_docstrings
class PandasOnRayDataframePartition(PandasDataframePartition):
    """
    The class implements the interface in ``PandasDataframePartition``.

    Parameters
    ----------
    data : ObjectIDType or DeferredExecution
        A reference to ``pandas.DataFrame`` that needs to be wrapped with this class
        or a reference to DeferredExecution that needs to be executed on demand.
    length : ObjectIDType or int, optional
        Length or reference to it of wrapped ``pandas.DataFrame``.
    width : ObjectIDType or int, optional
        Width or reference to it of wrapped ``pandas.DataFrame``.
    ip : ObjectIDType or str, optional
        Node IP address or reference to it that holds wrapped ``pandas.DataFrame``.
    meta : MetaList
        Meta information, containing the lengths and the worker address (the last value).
    meta_offset : int
        The lengths offset in the meta list.
    """
    execution_wrapper = RayWrapper

    def __init__(self, data: Union[ray.ObjectRef, 'ClientObjectRef', DeferredExecution], length: int=None, width: int=None, ip: str=None, meta: MetaList=None, meta_offset: int=0):
        super().__init__()
        if isinstance(data, DeferredExecution):
            data.subscribe()
        self._data_ref = data
        if meta is None:
            self._meta = MetaList([length, width, ip])
            self._meta_offset = 0
        else:
            self._meta = meta
            self._meta_offset = meta_offset
        log = get_logger()
        self._is_debug(log) and log.debug('Partition ID: {}, Height: {}, Width: {}, Node IP: {}'.format(self._identity, str(self._length_cache), str(self._width_cache), str(self._ip_cache)))

    @disable_logging
    def __del__(self):
        """Unsubscribe from DeferredExecution."""
        if isinstance(self._data_ref, DeferredExecution):
            self._data_ref.unsubscribe()

    def apply(self, func: Union[Callable, ray.ObjectRef], *args, **kwargs):
        """
        Apply a function to the object wrapped by this partition.

        Parameters
        ----------
        func : callable or ray.ObjectRef
            A function to apply.
        *args : iterable
            Additional positional arguments to be passed in `func`.
        **kwargs : dict
            Additional keyword arguments to be passed in `func`.

        Returns
        -------
        PandasOnRayDataframePartition
            A new ``PandasOnRayDataframePartition`` object.

        Notes
        -----
        It does not matter if `func` is callable or an ``ray.ObjectRef``. Ray will
        handle it correctly either way. The keyword arguments are sent as a dictionary.
        """
        log = get_logger()
        self._is_debug(log) and log.debug(f'ENTER::Partition.apply::{self._identity}')
        de = DeferredExecution(self._data_ref, func, args, kwargs)
        data, meta, meta_offset = de.exec()
        self._is_debug(log) and log.debug(f'EXIT::Partition.apply::{self._identity}')
        return self.__constructor__(data, meta=meta, meta_offset=meta_offset)

    @_inherit_docstrings(PandasDataframePartition.add_to_apply_calls)
    def add_to_apply_calls(self, func: Union[Callable, ray.ObjectRef], *args, length=None, width=None, **kwargs):
        return self.__constructor__(data=DeferredExecution(self._data_ref, func, args, kwargs), length=length, width=width)

    @_inherit_docstrings(PandasDataframePartition.drain_call_queue)
    def drain_call_queue(self):
        data = self._data_ref
        if not isinstance(data, DeferredExecution):
            return data
        log = get_logger()
        self._is_debug(log) and log.debug(f'ENTER::Partition.drain_call_queue::{self._identity}')
        self._data_ref, self._meta, self._meta_offset = data.exec()
        self._is_debug(log) and log.debug(f'EXIT::Partition.drain_call_queue::{self._identity}')

    @_inherit_docstrings(PandasDataframePartition.wait)
    def wait(self):
        self.drain_call_queue()
        RayWrapper.wait(self._data_ref)

    def __copy__(self):
        """
        Create a copy of this partition.

        Returns
        -------
        PandasOnRayDataframePartition
            A copy of this partition.
        """
        return self.__constructor__(self._data_ref, meta=self._meta, meta_offset=self._meta_offset)

    def mask(self, row_labels, col_labels):
        """
        Lazily create a mask that extracts the indices provided.

        Parameters
        ----------
        row_labels : list-like, slice or label
            The row labels for the rows to extract.
        col_labels : list-like, slice or label
            The column labels for the columns to extract.

        Returns
        -------
        PandasOnRayDataframePartition
            A new ``PandasOnRayDataframePartition`` object.
        """
        log = get_logger()
        self._is_debug(log) and log.debug(f'ENTER::Partition.mask::{self._identity}')
        new_obj = super().mask(row_labels, col_labels)
        if isinstance(row_labels, slice) and isinstance((len_cache := self._length_cache), ObjectIDType):
            if row_labels == slice(None):
                new_obj._length_cache = len_cache
            else:
                new_obj._length_cache = SlicerHook(len_cache, row_labels)
        if isinstance(col_labels, slice) and isinstance((width_cache := self._width_cache), ObjectIDType):
            if col_labels == slice(None):
                new_obj._width_cache = width_cache
            else:
                new_obj._width_cache = SlicerHook(width_cache, col_labels)
        self._is_debug(log) and log.debug(f'EXIT::Partition.mask::{self._identity}')
        return new_obj

    @classmethod
    def put(cls, obj: pandas.DataFrame):
        """
        Put the data frame into Plasma store and wrap it with partition object.

        Parameters
        ----------
        obj : pandas.DataFrame
            A data frame to be put.

        Returns
        -------
        PandasOnRayDataframePartition
            A new ``PandasOnRayDataframePartition`` object.
        """
        return cls(cls.execution_wrapper.put(obj), len(obj.index), len(obj.columns))

    @classmethod
    def preprocess_func(cls, func):
        """
        Put a function into the Plasma store to use in ``apply``.

        Parameters
        ----------
        func : callable
            A function to preprocess.

        Returns
        -------
        ray.ObjectRef
            A reference to `func`.
        """
        return cls.execution_wrapper.put(func)

    def length(self, materialize=True):
        """
        Get the length of the object wrapped by this partition.

        Parameters
        ----------
        materialize : bool, default: True
            Whether to forcibly materialize the result into an integer. If ``False``
            was specified, may return a future of the result if it hasn't been
            materialized yet.

        Returns
        -------
        int or ray.ObjectRef
            The length of the object.
        """
        if (length := self._length_cache) is None:
            self.drain_call_queue()
            if (length := self._length_cache) is None:
                length, self._width_cache = _get_index_and_columns.remote(self._data_ref)
                self._length_cache = length
        if materialize and isinstance(length, ObjectIDType):
            self._length_cache = length = RayWrapper.materialize(length)
        return length

    def width(self, materialize=True):
        """
        Get the width of the object wrapped by the partition.

        Parameters
        ----------
        materialize : bool, default: True
            Whether to forcibly materialize the result into an integer. If ``False``
            was specified, may return a future of the result if it hasn't been
            materialized yet.

        Returns
        -------
        int or ray.ObjectRef
            The width of the object.
        """
        if (width := self._width_cache) is None:
            self.drain_call_queue()
            if (width := self._width_cache) is None:
                self._length_cache, width = _get_index_and_columns.remote(self._data_ref)
                self._width_cache = width
        if materialize and isinstance(width, ObjectIDType):
            self._width_cache = width = RayWrapper.materialize(width)
        return width

    def ip(self, materialize=True):
        """
        Get the node IP address of the object wrapped by this partition.

        Parameters
        ----------
        materialize : bool, default: True
            Whether to forcibly materialize the result into an integer. If ``False``
            was specified, may return a future of the result if it hasn't been
            materialized yet.

        Returns
        -------
        str
            IP address of the node that holds the data.
        """
        if (ip := self._ip_cache) is None:
            self.drain_call_queue()
        if materialize and isinstance(ip, ObjectIDType):
            self._ip_cache = ip = RayWrapper.materialize(ip)
        return ip

    @property
    def _data(self) -> Union[ray.ObjectRef, 'ClientObjectRef']:
        self.drain_call_queue()
        return self._data_ref

    @property
    def _length_cache(self):
        return self._meta[self._meta_offset]

    @_length_cache.setter
    def _length_cache(self, value):
        self._meta[self._meta_offset] = value

    @property
    def _width_cache(self):
        return self._meta[self._meta_offset + 1]

    @_width_cache.setter
    def _width_cache(self, value):
        self._meta[self._meta_offset + 1] = value

    @property
    def _ip_cache(self):
        return self._meta[-1]

    @_ip_cache.setter
    def _ip_cache(self, value):
        self._meta[-1] = value