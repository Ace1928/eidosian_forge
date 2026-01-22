import abc
import ctypes
import inspect
import json
import warnings
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from os import SEEK_END, environ
from os.path import getsize
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import scipy.sparse
from .compat import (PANDAS_INSTALLED, PYARROW_INSTALLED, arrow_cffi, arrow_is_floating, arrow_is_integer, concat,
from .libpath import find_lib_path
def __inner_predict(self, data_idx: int) -> np.ndarray:
    """Predict for training and validation dataset."""
    if data_idx >= self.__num_dataset:
        raise ValueError('Data_idx should be smaller than number of dataset')
    if self.__inner_predict_buffer[data_idx] is None:
        if data_idx == 0:
            n_preds = self.train_set.num_data() * self.__num_class
        else:
            n_preds = self.valid_sets[data_idx - 1].num_data() * self.__num_class
        self.__inner_predict_buffer[data_idx] = np.empty(n_preds, dtype=np.float64)
    if not self.__is_predicted_cur_iter[data_idx]:
        tmp_out_len = ctypes.c_int64(0)
        data_ptr = self.__inner_predict_buffer[data_idx].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        _safe_call(_LIB.LGBM_BoosterGetPredict(self._handle, ctypes.c_int(data_idx), ctypes.byref(tmp_out_len), data_ptr))
        if tmp_out_len.value != len(self.__inner_predict_buffer[data_idx]):
            raise ValueError(f'Wrong length of predict results for data {data_idx}')
        self.__is_predicted_cur_iter[data_idx] = True
    result: np.ndarray = self.__inner_predict_buffer[data_idx]
    if self.__num_class > 1:
        num_data = result.size // self.__num_class
        result = result.reshape(num_data, self.__num_class, order='F')
    return result