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
class NDArrayIter(DataIter):
    """Returns an iterator for ``mx.nd.NDArray``, ``numpy.ndarray``, ``h5py.Dataset``
    ``mx.nd.sparse.CSRNDArray`` or ``scipy.sparse.csr_matrix``.

    Examples
    --------
    >>> data = np.arange(40).reshape((10,2,2))
    >>> labels = np.ones([10, 1])
    >>> dataiter = mx.io.NDArrayIter(data, labels, 3, True, last_batch_handle='discard')
    >>> for batch in dataiter:
    ...     print batch.data[0].asnumpy()
    ...     batch.data[0].shape
    ...
    [[[ 36.  37.]
      [ 38.  39.]]
     [[ 16.  17.]
      [ 18.  19.]]
     [[ 12.  13.]
      [ 14.  15.]]]
    (3L, 2L, 2L)
    [[[ 32.  33.]
      [ 34.  35.]]
     [[  4.   5.]
      [  6.   7.]]
     [[ 24.  25.]
      [ 26.  27.]]]
    (3L, 2L, 2L)
    [[[  8.   9.]
      [ 10.  11.]]
     [[ 20.  21.]
      [ 22.  23.]]
     [[ 28.  29.]
      [ 30.  31.]]]
    (3L, 2L, 2L)
    >>> dataiter.provide_data # Returns a list of `DataDesc`
    [DataDesc[data,(3, 2L, 2L),<type 'numpy.float32'>,NCHW]]
    >>> dataiter.provide_label # Returns a list of `DataDesc`
    [DataDesc[softmax_label,(3, 1L),<type 'numpy.float32'>,NCHW]]

    In the above example, data is shuffled as `shuffle` parameter is set to `True`
    and remaining examples are discarded as `last_batch_handle` parameter is set to `discard`.

    Usage of `last_batch_handle` parameter:

    >>> dataiter = mx.io.NDArrayIter(data, labels, 3, True, last_batch_handle='pad')
    >>> batchidx = 0
    >>> for batch in dataiter:
    ...     batchidx += 1
    ...
    >>> batchidx  # Padding added after the examples read are over. So, 10/3+1 batches are created.
    4
    >>> dataiter = mx.io.NDArrayIter(data, labels, 3, True, last_batch_handle='discard')
    >>> batchidx = 0
    >>> for batch in dataiter:
    ...     batchidx += 1
    ...
    >>> batchidx # Remaining examples are discarded. So, 10/3 batches are created.
    3
    >>> dataiter = mx.io.NDArrayIter(data, labels, 3, False, last_batch_handle='roll_over')
    >>> batchidx = 0
    >>> for batch in dataiter:
    ...     batchidx += 1
    ...
    >>> batchidx # Remaining examples are rolled over to the next iteration.
    3
    >>> dataiter.reset()
    >>> dataiter.next().data[0].asnumpy()
    [[[ 36.  37.]
      [ 38.  39.]]
     [[ 0.  1.]
      [ 2.  3.]]
     [[ 4.  5.]
      [ 6.  7.]]]
    (3L, 2L, 2L)

    `NDArrayIter` also supports multiple input and labels.

    >>> data = {'data1':np.zeros(shape=(10,2,2)), 'data2':np.zeros(shape=(20,2,2))}
    >>> label = {'label1':np.zeros(shape=(10,1)), 'label2':np.zeros(shape=(20,1))}
    >>> dataiter = mx.io.NDArrayIter(data, label, 3, True, last_batch_handle='discard')

    `NDArrayIter` also supports ``mx.nd.sparse.CSRNDArray``
    with `last_batch_handle` set to `discard`.

    >>> csr_data = mx.nd.array(np.arange(40).reshape((10,4))).tostype('csr')
    >>> labels = np.ones([10, 1])
    >>> dataiter = mx.io.NDArrayIter(csr_data, labels, 3, last_batch_handle='discard')
    >>> [batch.data[0] for batch in dataiter]
    [
    <CSRNDArray 3x4 @cpu(0)>,
    <CSRNDArray 3x4 @cpu(0)>,
    <CSRNDArray 3x4 @cpu(0)>]

    Parameters
    ----------
    data: array or list of array or dict of string to array
        The input data.
    label: array or list of array or dict of string to array, optional
        The input label.
    batch_size: int
        Batch size of data.
    shuffle: bool, optional
        Whether to shuffle the data.
        Only supported if no h5py.Dataset inputs are used.
    last_batch_handle : str, optional
        How to handle the last batch. This parameter can be 'pad', 'discard' or
        'roll_over'.
        If 'pad', the last batch will be padded with data starting from the begining
        If 'discard', the last batch will be discarded
        If 'roll_over', the remaining elements will be rolled over to the next iteration and
        note that it is intended for training and can cause problems if used for prediction.
    data_name : str, optional
        The data name.
    label_name : str, optional
        The label name.
    layout : str, optional
        The data layout
    """

    def __init__(self, data, label=None, batch_size=1, shuffle=False, last_batch_handle='pad', data_name='data', label_name='softmax_label', layout='NCHW'):
        super(NDArrayIter, self).__init__(batch_size)
        self.data = _init_data(data, allow_empty=False, default_name=data_name)
        self.label = _init_data(label, allow_empty=True, default_name=label_name)
        if (_has_instance(self.data, CSRNDArray) or _has_instance(self.label, CSRNDArray)) and last_batch_handle != 'discard':
            raise NotImplementedError('`NDArrayIter` only supports ``CSRNDArray`` with `last_batch_handle` set to `discard`.')
        self.idx = np.arange(self.data[0][1].shape[0])
        self.shuffle = shuffle
        self.last_batch_handle = last_batch_handle
        self.batch_size = batch_size
        self.cursor = -self.batch_size
        self.num_data = self.idx.shape[0]
        self.reset()
        self.data_list = [x[1] for x in self.data] + [x[1] for x in self.label]
        self.num_source = len(self.data_list)
        self._cache_data = None
        self._cache_label = None
        self.layout = layout

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator."""
        batch_axis = self.layout.find('N')
        return [DataDesc(k, tuple(list(v.shape[:batch_axis]) + [self.batch_size] + list(v.shape[batch_axis + 1:])), v.dtype, layout=self.layout) for k, v in self.data]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator."""
        batch_axis = self.layout.find('N')
        return [DataDesc(k, tuple(list(v.shape[:batch_axis]) + [self.batch_size] + list(v.shape[batch_axis + 1:])), v.dtype, layout=self.layout) for k, v in self.label]

    def hard_reset(self):
        """Ignore roll over data and set to start."""
        if self.shuffle:
            self._shuffle_data()
        self.cursor = -self.batch_size
        self._cache_data = None
        self._cache_label = None

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        if self.shuffle:
            self._shuffle_data()
        if self.last_batch_handle == 'roll_over' and self.num_data - self.batch_size < self.cursor < self.num_data:
            self.cursor = self.cursor - self.num_data - self.batch_size
        else:
            self.cursor = -self.batch_size

    def iter_next(self):
        """Increments the coursor by batch_size for next batch
        and check current cursor if it exceed the number of data points."""
        self.cursor += self.batch_size
        return self.cursor < self.num_data

    def next(self):
        """Returns the next batch of data."""
        if not self.iter_next():
            raise StopIteration
        data = self.getdata()
        label = self.getlabel()
        if data[0].shape[self.layout.find('N')] != self.batch_size:
            self._cache_data = data
            self._cache_label = label
            raise StopIteration
        return DataBatch(data=data, label=label, pad=self.getpad(), index=None)

    def _getdata(self, data_source, start=None, end=None):
        """Load data from underlying arrays."""
        assert start is not None or end is not None, 'should at least specify start or end'
        start = start if start is not None else 0
        if end is None:
            end = data_source[0][1].shape[0] if data_source else 0
        s = slice(start, end)
        return [_slice_along_batch_axis(x[1], s, self.layout.find('N')) if isinstance(x[1], (np.ndarray, NDArray)) else array(x[1][sorted(self.idx[s])][[list(self.idx[s]).index(i) for i in sorted(self.idx[s])]]) for x in data_source]

    def _concat(self, first_data, second_data):
        """Helper function to concat two NDArrays."""
        if not first_data or not second_data:
            return first_data if first_data else second_data
        assert len(first_data) == len(second_data), 'data source should contain the same size'
        return [concat(first_data[i], second_data[i], dim=self.layout.find('N')) for i in range(len(first_data))]

    def _tile(self, data, repeats):
        if not data:
            return []
        res = []
        for datum in data:
            reps = [1] * len(datum.shape)
            reps[0] = repeats
            res.append(tile(datum, reps))
        return res

    def _batchify(self, data_source):
        """Load data from underlying arrays, internal use only."""
        assert self.cursor < self.num_data, 'DataIter needs reset.'
        if self.last_batch_handle == 'roll_over' and -self.batch_size < self.cursor < 0:
            assert self._cache_data is not None or self._cache_label is not None, 'next epoch should have cached data'
            cache_data = self._cache_data if self._cache_data is not None else self._cache_label
            second_data = self._getdata(data_source, end=self.cursor + self.batch_size)
            if self._cache_data is not None:
                self._cache_data = None
            else:
                self._cache_label = None
            return self._concat(cache_data, second_data)
        elif self.last_batch_handle == 'pad' and self.cursor + self.batch_size > self.num_data:
            pad = self.batch_size - self.num_data + self.cursor
            first_data = self._getdata(data_source, start=self.cursor)
            if pad > self.num_data:
                repeats = pad // self.num_data
                second_data = self._tile(self._getdata(data_source, end=self.num_data), repeats)
                if pad % self.num_data != 0:
                    second_data = self._concat(second_data, self._getdata(data_source, end=pad % self.num_data))
            else:
                second_data = self._getdata(data_source, end=pad)
            return self._concat(first_data, second_data)
        else:
            if self.cursor + self.batch_size < self.num_data:
                end_idx = self.cursor + self.batch_size
            else:
                end_idx = self.num_data
            return self._getdata(data_source, self.cursor, end_idx)

    def getdata(self):
        """Get data."""
        return self._batchify(self.data)

    def getlabel(self):
        """Get label."""
        return self._batchify(self.label)

    def getpad(self):
        """Get pad value of DataBatch."""
        if self.last_batch_handle == 'pad' and self.cursor + self.batch_size > self.num_data:
            return self.cursor + self.batch_size - self.num_data
        elif self.last_batch_handle == 'roll_over' and -self.batch_size < self.cursor < 0:
            return -self.cursor
        else:
            return 0

    def _shuffle_data(self):
        """Shuffle the data."""
        np.random.shuffle(self.idx)
        self.data = _getdata_by_idx(self.data, self.idx)
        self.label = _getdata_by_idx(self.label, self.idx)