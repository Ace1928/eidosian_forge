from __future__ import annotations
import pickle as pkl
import re
import warnings
from typing import TYPE_CHECKING, Any, Hashable, Literal, Optional, Sequence, Union
import numpy as np
import pandas
import pandas.core.generic
import pandas.core.resample
import pandas.core.window.rolling
from pandas._libs import lib
from pandas._libs.tslibs import to_offset
from pandas._typing import (
from pandas.compat import numpy as numpy_compat
from pandas.core.common import count_not_none, pipe
from pandas.core.dtypes.common import (
from pandas.core.indexes.api import ensure_index
from pandas.core.methods.describe import _refine_percentiles
from pandas.util._validators import (
from modin import pandas as pd
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger, disable_logging
from modin.pandas.accessor import CachedAccessor, ModinAPI
from modin.pandas.utils import is_scalar
from modin.utils import _inherit_docstrings, expanduser_path_arg, try_cast_to_pandas
from .utils import _doc_binary_op, is_full_grab_slice
def _ensure_index(self, index_like, axis=0):
    """
        Ensure that we have an index from some index-like object.
        """
    if self._query_compiler.has_multiindex(axis=axis) and (not isinstance(index_like, pandas.Index)) and is_list_like(index_like) and (len(index_like) > 0) and isinstance(index_like[0], tuple):
        try:
            return pandas.MultiIndex.from_tuples(index_like)
        except TypeError:
            pass
    return ensure_index(index_like)