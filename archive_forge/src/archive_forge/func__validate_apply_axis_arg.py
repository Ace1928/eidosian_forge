from __future__ import annotations
from contextlib import contextmanager
import copy
from functools import partial
import operator
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
import pandas as pd
from pandas import (
import pandas.core.common as com
from pandas.core.frame import (
from pandas.core.generic import NDFrame
from pandas.core.shared_docs import _shared_docs
from pandas.io.formats.format import save_to_buffer
from pandas.io.formats.style_render import (
def _validate_apply_axis_arg(arg: NDFrame | Sequence | np.ndarray, arg_name: str, dtype: Any | None, data: NDFrame) -> np.ndarray:
    """
    For the apply-type methods, ``axis=None`` creates ``data`` as DataFrame, and for
    ``axis=[1,0]`` it creates a Series. Where ``arg`` is expected as an element
    of some operator with ``data`` we must make sure that the two are compatible shapes,
    or raise.

    Parameters
    ----------
    arg : sequence, Series or DataFrame
        the user input arg
    arg_name : string
        name of the arg for use in error messages
    dtype : numpy dtype, optional
        forced numpy dtype if given
    data : Series or DataFrame
        underling subset of Styler data on which operations are performed

    Returns
    -------
    ndarray
    """
    dtype = {'dtype': dtype} if dtype else {}
    if isinstance(arg, Series) and isinstance(data, DataFrame):
        raise ValueError(f"'{arg_name}' is a Series but underlying data for operations is a DataFrame since 'axis=None'")
    if isinstance(arg, DataFrame) and isinstance(data, Series):
        raise ValueError(f"'{arg_name}' is a DataFrame but underlying data for operations is a Series with 'axis in [0,1]'")
    if isinstance(arg, (Series, DataFrame)):
        arg = arg.reindex_like(data, method=None).to_numpy(**dtype)
    else:
        arg = np.asarray(arg, **dtype)
        assert isinstance(arg, np.ndarray)
        if arg.shape != data.shape:
            raise ValueError(f"supplied '{arg_name}' is not correct shape for data over selected 'axis': got {arg.shape}, expected {data.shape}")
    return arg