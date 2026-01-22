import functools
import pickle
from typing import Dict, Callable, Optional, TypeVar, Generic, Iterator
from torch.utils.data.datapipes._typing import _DataPipeMeta, _IterDataPipeMeta
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.utils.common import (
from torch.utils.data.dataset import Dataset, IterableDataset
class MapDataPipe(Dataset[T_co], metaclass=_DataPipeMeta):
    """
    Map-style DataPipe.

    All datasets that represent a map from keys to data samples should subclass this.
    Subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given, unique key. Subclasses can also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.

    These DataPipes can be invoked in two ways, using the class constructor or applying their
    functional form onto an existing `MapDataPipe` (recommend, available to most but not all DataPipes).

    Note:
        :class:`~torch.utils.data.DataLoader` by default constructs an index
        sampler that yields integral indices. To make it work with a map-style
        DataPipe with non-integral indices/keys, a custom sampler must be provided.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper, Mapper
        >>> dp = SequenceWrapper(range(10))
        >>> map_dp_1 = dp.map(lambda x: x + 1)  # Using functional form (recommended)
        >>> list(map_dp_1)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> map_dp_2 = Mapper(dp, lambda x: x + 1)  # Using class constructor
        >>> list(map_dp_2)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> batch_dp = map_dp_1.batch(batch_size=2)
        >>> list(batch_dp)
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    """
    functions: Dict[str, Callable] = {}
    reduce_ex_hook: Optional[Callable] = None
    getstate_hook: Optional[Callable] = None
    str_hook: Optional[Callable] = None
    repr_hook: Optional[Callable] = None

    def __getattr__(self, attribute_name):
        if attribute_name in MapDataPipe.functions:
            if attribute_name in _map_deprecated_functional_names:
                kwargs = _map_deprecated_functional_names[attribute_name]
                _deprecation_warning(**kwargs)
            f = MapDataPipe.functions[attribute_name]
            function = functools.partial(f, self)
            functools.update_wrapper(wrapper=function, wrapped=f, assigned=('__doc__',))
            return function
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attribute_name}")

    @classmethod
    def register_function(cls, function_name, function):
        cls.functions[function_name] = function

    @classmethod
    def register_datapipe_as_function(cls, function_name, cls_to_register):
        if function_name in cls.functions:
            raise Exception(f'Unable to add DataPipe function name {function_name} as it is already taken')

        def class_function(cls, source_dp, *args, **kwargs):
            result_pipe = cls(source_dp, *args, **kwargs)
            return result_pipe
        function = functools.partial(class_function, cls_to_register)
        functools.update_wrapper(wrapper=function, wrapped=cls_to_register, assigned=('__doc__',))
        cls.functions[function_name] = function

    def __getstate__(self):
        """
        Serialize `lambda` functions when `dill` is available.

        If this doesn't cover your custom DataPipe's use case, consider writing custom methods for
        `__getstate__` and `__setstate__`, or use `pickle.dumps` for serialization.
        """
        state = self.__dict__
        if MapDataPipe.getstate_hook is not None:
            return MapDataPipe.getstate_hook(state)
        return state

    def __reduce_ex__(self, *args, **kwargs):
        if MapDataPipe.reduce_ex_hook is not None:
            try:
                return MapDataPipe.reduce_ex_hook(self)
            except NotImplementedError:
                pass
        return super().__reduce_ex__(*args, **kwargs)

    @classmethod
    def set_getstate_hook(cls, hook_fn):
        if MapDataPipe.getstate_hook is not None and hook_fn is not None:
            raise Exception('Attempt to override existing getstate_hook')
        MapDataPipe.getstate_hook = hook_fn

    @classmethod
    def set_reduce_ex_hook(cls, hook_fn):
        if MapDataPipe.reduce_ex_hook is not None and hook_fn is not None:
            raise Exception('Attempt to override existing reduce_ex_hook')
        MapDataPipe.reduce_ex_hook = hook_fn

    def __repr__(self):
        if self.repr_hook is not None:
            return self.repr_hook(self)
        return str(self.__class__.__qualname__)

    def __str__(self):
        if self.str_hook is not None:
            return self.str_hook(self)
        return str(self.__class__.__qualname__)

    def __dir__(self):
        return list(super().__dir__()) + list(self.functions.keys())