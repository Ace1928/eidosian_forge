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
def _validate_function(self, func, on_invalid=None):
    """
        Check the validity of the function which is intended to be applied to the frame.

        Parameters
        ----------
        func : object
        on_invalid : callable(str, cls), optional
            Function to call in case invalid `func` is met, `on_invalid` takes an error
            message and an exception type as arguments. If not specified raise an
            appropriate exception.
            **Note:** This parameter is a hack to concord with pandas error types.
        """

    def error_raiser(msg, exception=Exception):
        raise exception(msg)
    if on_invalid is None:
        on_invalid = error_raiser
    if isinstance(func, dict):
        [self._validate_function(fn, on_invalid) for fn in func.values()]
        return
    if not is_list_like(func):
        func = [func]
    for fn in func:
        if isinstance(fn, str):
            if not (hasattr(self, fn) or hasattr(np, fn)):
                on_invalid(f"'{fn}' is not a valid function for '{type(self).__name__}' object", AttributeError)
        elif not callable(fn):
            on_invalid(f'One of the passed functions has an invalid type: {type(fn)}: {fn}, ' + 'only callable or string is acceptable.', TypeError)