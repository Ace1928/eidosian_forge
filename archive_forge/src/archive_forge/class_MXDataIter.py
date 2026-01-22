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
class MXDataIter(DataIter):
    """A python wrapper a C++ data iterator.

    This iterator is the Python wrapper to all native C++ data iterators, such
    as `CSVIter`, `ImageRecordIter`, `MNISTIter`, etc. When initializing
    `CSVIter` for example, you will get an `MXDataIter` instance to use in your
    Python code. Calls to `next`, `reset`, etc will be delegated to the
    underlying C++ data iterators.

    Usually you don't need to interact with `MXDataIter` directly unless you are
    implementing your own data iterators in C++. To do that, please refer to
    examples under the `src/io` folder.

    Parameters
    ----------
    handle : DataIterHandle, required
        The handle to the underlying C++ Data Iterator.
    data_name : str, optional
        Data name. Default to "data".
    label_name : str, optional
        Label name. Default to "softmax_label".

    See Also
    --------
    src/io : The underlying C++ data iterator implementation, e.g., `CSVIter`.
    """

    def __init__(self, handle, data_name='data', label_name='softmax_label', **_):
        super(MXDataIter, self).__init__()
        self.handle = handle
        self._debug_skip_load = False
        self.first_batch = None
        self.first_batch = self.next()
        data = self.first_batch.data[0]
        label = self.first_batch.label[0]
        self.provide_data = [DataDesc(data_name, data.shape, data.dtype)]
        self.provide_label = [DataDesc(label_name, label.shape, label.dtype)]
        self.batch_size = data.shape[0]

    def __del__(self):
        check_call(_LIB.MXDataIterFree(self.handle))

    def debug_skip_load(self):
        self._debug_skip_load = True
        logging.info('Set debug_skip_load to be true, will simply return first batch')

    def reset(self):
        self._debug_at_begin = True
        self.first_batch = None
        check_call(_LIB.MXDataIterBeforeFirst(self.handle))

    def next(self):
        if self._debug_skip_load and (not self._debug_at_begin):
            return DataBatch(data=[self.getdata()], label=[self.getlabel()], pad=self.getpad(), index=self.getindex())
        if self.first_batch is not None:
            batch = self.first_batch
            self.first_batch = None
            return batch
        self._debug_at_begin = False
        next_res = ctypes.c_int(0)
        check_call(_LIB.MXDataIterNext(self.handle, ctypes.byref(next_res)))
        if next_res.value:
            return DataBatch(data=[self.getdata()], label=[self.getlabel()], pad=self.getpad(), index=self.getindex())
        else:
            raise StopIteration

    def iter_next(self):
        if self.first_batch is not None:
            return True
        next_res = ctypes.c_int(0)
        check_call(_LIB.MXDataIterNext(self.handle, ctypes.byref(next_res)))
        return next_res.value

    def getdata(self):
        hdl = NDArrayHandle()
        check_call(_LIB.MXDataIterGetData(self.handle, ctypes.byref(hdl)))
        return _ndarray_cls(hdl, False)

    def getlabel(self):
        hdl = NDArrayHandle()
        check_call(_LIB.MXDataIterGetLabel(self.handle, ctypes.byref(hdl)))
        return _ndarray_cls(hdl, False)

    def getindex(self):
        index_size = ctypes.c_uint64(0)
        index_data = ctypes.POINTER(ctypes.c_uint64)()
        check_call(_LIB.MXDataIterGetIndex(self.handle, ctypes.byref(index_data), ctypes.byref(index_size)))
        if index_size.value:
            address = ctypes.addressof(index_data.contents)
            dbuffer = (ctypes.c_uint64 * index_size.value).from_address(address)
            np_index = np.frombuffer(dbuffer, dtype=np.uint64)
            return np_index.copy()
        else:
            return None

    def getpad(self):
        pad = ctypes.c_int(0)
        check_call(_LIB.MXDataIterGetPadNum(self.handle, ctypes.byref(pad)))
        return pad.value