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
def __inner_eval(self, data_name: str, data_idx: int, feval: Optional[Union[_LGBM_CustomEvalFunction, List[_LGBM_CustomEvalFunction]]]) -> List[_LGBM_BoosterEvalMethodResultType]:
    """Evaluate training or validation data."""
    if data_idx >= self.__num_dataset:
        raise ValueError('Data_idx should be smaller than number of dataset')
    self.__get_eval_info()
    ret = []
    if self.__num_inner_eval > 0:
        result = np.empty(self.__num_inner_eval, dtype=np.float64)
        tmp_out_len = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterGetEval(self._handle, ctypes.c_int(data_idx), ctypes.byref(tmp_out_len), result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
        if tmp_out_len.value != self.__num_inner_eval:
            raise ValueError('Wrong length of eval results')
        for i in range(self.__num_inner_eval):
            ret.append((data_name, self.__name_inner_eval[i], result[i], self.__higher_better_inner_eval[i]))
    if callable(feval):
        feval = [feval]
    if feval is not None:
        if data_idx == 0:
            cur_data = self.train_set
        else:
            cur_data = self.valid_sets[data_idx - 1]
        for eval_function in feval:
            if eval_function is None:
                continue
            feval_ret = eval_function(self.__inner_predict(data_idx), cur_data)
            if isinstance(feval_ret, list):
                for eval_name, val, is_higher_better in feval_ret:
                    ret.append((data_name, eval_name, val, is_higher_better))
            else:
                eval_name, val, is_higher_better = feval_ret
                ret.append((data_name, eval_name, val, is_higher_better))
    return ret