import ctypes
import logging
import os
import shutil
import warnings
import numpy as np
from ..base import _LIB, check_call, py_str
from ..base import c_array, c_str, mx_uint, c_str_array
from ..base import NDArrayHandle, SymbolHandle
from ..symbol import Symbol
from ..symbol import load as sym_load
from .. import ndarray
from ..ndarray import load as nd_load
from ..ndarray import save as nd_save
from ..ndarray import NDArray
from ..io import DataIter, DataDesc, DataBatch
from ..context import cpu, Context
from ..module import Module
class _DataIterWrapper(DataIter):
    """DataIter wrapper for general iterator, e.g., gluon dataloader"""

    def __init__(self, calib_data):
        self._data = calib_data
        try:
            calib_iter = iter(calib_data)
        except TypeError as e:
            raise TypeError('calib_data is not a valid iterator. {}'.format(str(e)))
        data_example = next(calib_iter)
        if isinstance(data_example, (list, tuple)):
            data_example = list(data_example)
        else:
            data_example = [data_example]
        num_data = len(data_example)
        assert num_data > 0
        if len(data_example[0].shape) > 4:
            data_example[0] = data_example[0].reshape((-1,) + data_example[0].shape[2:])
        self.provide_data = [DataDesc(name='data', shape=data_example[0].shape)]
        self.provide_data += [DataDesc(name='data{}'.format(i), shape=x.shape) for i, x in enumerate(data_example[1:])]
        if num_data >= 3:
            self.provide_data = [DataDesc(name='data{}'.format(i), shape=x.shape) for i, x in enumerate(data_example[0:])]
        self.batch_size = data_example[0].shape[0]
        self.reset()

    def reset(self):
        self._iter = iter(self._data)

    def next(self):
        next_data = next(self._iter)
        if len(next_data[0].shape) > 4:
            next_data[0] = next_data[0].reshape((-1,) + next_data[0].shape[2:])
        return DataBatch(data=next_data)