import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _convert_unknown_data(data: DataType) -> DataType:
    warnings.warn(f'Unknown data type: {type(data)}, trying to convert it to csr_matrix', UserWarning)
    try:
        import scipy.sparse
    except ImportError:
        return None
    try:
        data = scipy.sparse.csr_matrix(data)
    except Exception:
        return None
    return data