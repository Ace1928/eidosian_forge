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
class _ArrowCArray:
    """Simple wrapper around the C representation of an Arrow type."""
    n_chunks: int
    chunks: arrow_cffi.CData
    schema: arrow_cffi.CData

    def __init__(self, n_chunks: int, chunks: arrow_cffi.CData, schema: arrow_cffi.CData):
        self.n_chunks = n_chunks
        self.chunks = chunks
        self.schema = schema

    @property
    def chunks_ptr(self) -> int:
        """Returns the address of the pointer to the list of chunks making up the array."""
        return int(arrow_cffi.cast('uintptr_t', arrow_cffi.addressof(self.chunks[0])))

    @property
    def schema_ptr(self) -> int:
        """Returns the address of the pointer to the schema of the array."""
        return int(arrow_cffi.cast('uintptr_t', self.schema))