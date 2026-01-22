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
def _init_from_ref_dataset(self, total_nrow: int, ref_dataset: _DatasetHandle) -> 'Dataset':
    """Create dataset from a reference dataset.

        Parameters
        ----------
        total_nrow : int
            Number of rows expected to add to dataset.
        ref_dataset : object
            Handle of reference dataset to extract metadata from.

        Returns
        -------
        self : Dataset
            Constructed Dataset object.
        """
    self._handle = ctypes.c_void_p()
    _safe_call(_LIB.LGBM_DatasetCreateByReference(ref_dataset, ctypes.c_int64(total_nrow), ctypes.byref(self._handle)))
    return self