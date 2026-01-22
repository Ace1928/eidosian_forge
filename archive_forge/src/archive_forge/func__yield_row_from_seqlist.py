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
@staticmethod
def _yield_row_from_seqlist(seqs: List[Sequence], indices: Iterable[int]):
    offset = 0
    seq_id = 0
    seq = seqs[seq_id]
    for row_id in indices:
        assert row_id >= offset, 'sample indices are expected to be monotonic'
        while row_id >= offset + len(seq):
            offset += len(seq)
            seq_id += 1
            seq = seqs[seq_id]
        id_in_seq = row_id - offset
        row = seq[id_in_seq]
        yield (row if row.flags['OWNDATA'] else row.copy())