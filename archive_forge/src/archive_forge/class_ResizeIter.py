from collections import namedtuple
import sys
import ctypes
import logging
import threading
import numpy as np
from ..base import _LIB
from ..base import c_str_array, mx_uint, py_str
from ..base import DataIterHandle, NDArrayHandle
from ..base import mx_real_t
from ..base import check_call, build_param_doc as _build_param_doc
from ..ndarray import NDArray
from ..ndarray.sparse import CSRNDArray
from ..ndarray import _ndarray_cls
from ..ndarray import array
from ..ndarray import concat, tile
from .utils import _init_data, _has_instance, _getdata_by_idx, _slice_along_batch_axis
class ResizeIter(DataIter):
    """Resize a data iterator to a given number of batches.

    Parameters
    ----------
    data_iter : DataIter
        The data iterator to be resized.
    size : int
        The number of batches per epoch to resize to.
    reset_internal : bool
        Whether to reset internal iterator on ResizeIter.reset.


    Examples
    --------
    >>> nd_iter = mx.io.NDArrayIter(mx.nd.ones((100,10)), batch_size=25)
    >>> resize_iter = mx.io.ResizeIter(nd_iter, 2)
    >>> for batch in resize_iter:
    ...     print(batch.data)
    [<NDArray 25x10 @cpu(0)>]
    [<NDArray 25x10 @cpu(0)>]
    """

    def __init__(self, data_iter, size, reset_internal=True):
        super(ResizeIter, self).__init__()
        self.data_iter = data_iter
        self.size = size
        self.reset_internal = reset_internal
        self.cur = 0
        self.current_batch = None
        self.provide_data = data_iter.provide_data
        self.provide_label = data_iter.provide_label
        self.batch_size = data_iter.batch_size
        if hasattr(data_iter, 'default_bucket_key'):
            self.default_bucket_key = data_iter.default_bucket_key

    def reset(self):
        self.cur = 0
        if self.reset_internal:
            self.data_iter.reset()

    def iter_next(self):
        if self.cur == self.size:
            return False
        try:
            self.current_batch = self.data_iter.next()
        except StopIteration:
            self.data_iter.reset()
            self.current_batch = self.data_iter.next()
        self.cur += 1
        return True

    def getdata(self):
        return self.current_batch.data

    def getlabel(self):
        return self.current_batch.label

    def getindex(self):
        return self.current_batch.index

    def getpad(self):
        return self.current_batch.pad