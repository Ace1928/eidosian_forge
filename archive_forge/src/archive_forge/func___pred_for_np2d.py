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
def __pred_for_np2d(self, mat: np.ndarray, start_iteration: int, num_iteration: int, predict_type: int) -> Tuple[np.ndarray, int]:
    """Predict for a 2-D numpy matrix."""
    if len(mat.shape) != 2:
        raise ValueError('Input numpy.ndarray or list must be 2 dimensional')
    nrow = mat.shape[0]
    if nrow > _MAX_INT32:
        sections = np.arange(start=_MAX_INT32, stop=nrow, step=_MAX_INT32)
        n_preds = [self.__get_num_preds(start_iteration, num_iteration, i, predict_type) for i in np.diff([0] + list(sections) + [nrow])]
        n_preds_sections = np.array([0] + n_preds, dtype=np.intp).cumsum()
        preds = np.empty(sum(n_preds), dtype=np.float64)
        for chunk, (start_idx_pred, end_idx_pred) in zip(np.array_split(mat, sections), zip(n_preds_sections, n_preds_sections[1:])):
            self.__inner_predict_np2d(mat=chunk, start_iteration=start_iteration, num_iteration=num_iteration, predict_type=predict_type, preds=preds[start_idx_pred:end_idx_pred])
        return (preds, nrow)
    else:
        return self.__inner_predict_np2d(mat=mat, start_iteration=start_iteration, num_iteration=num_iteration, predict_type=predict_type, preds=None)