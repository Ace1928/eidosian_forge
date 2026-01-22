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
def __init_from_seqs(self, seqs: List[Sequence], ref_dataset: Optional[_DatasetHandle]) -> 'Dataset':
    """
        Initialize data from list of Sequence objects.

        Sequence: Generic Data Access Object
            Supports random access and access by batch if properly defined by user

        Data scheme uniformity are trusted, not checked
        """
    total_nrow = sum((len(seq) for seq in seqs))
    if ref_dataset is not None:
        self._init_from_ref_dataset(total_nrow, ref_dataset)
    else:
        param_str = _param_dict_to_str(self.get_params())
        sample_cnt = _get_sample_count(total_nrow, param_str)
        sample_data, col_indices = self.__sample(seqs, total_nrow)
        self._init_from_sample(sample_data, col_indices, sample_cnt, total_nrow)
    for seq in seqs:
        nrow = len(seq)
        batch_size = getattr(seq, 'batch_size', None) or Sequence.batch_size
        for start in range(0, nrow, batch_size):
            end = min(start + batch_size, nrow)
            self._push_rows(seq[start:end])
    return self