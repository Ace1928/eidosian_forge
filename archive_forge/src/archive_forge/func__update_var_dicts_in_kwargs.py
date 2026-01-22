from __future__ import annotations
import datetime
import functools
import itertools
import os
import re
import sys
import warnings
from typing import (
import numpy as np
import pandas
from pandas._libs import lib
from pandas._typing import (
from pandas.core.common import apply_if_callable, get_cython_func
from pandas.core.computation.eval import _check_engine
from pandas.core.dtypes.common import (
from pandas.core.indexes.frozen import FrozenList
from pandas.io.formats.info import DataFrameInfo
from pandas.util._validators import validate_bool_kwarg
from modin.config import PersistentPickle
from modin.error_message import ErrorMessage
from modin.logging import disable_logging
from modin.pandas import Categorical
from modin.pandas.io import from_non_pandas, from_pandas, to_pandas
from modin.utils import (
from .accessor import CachedAccessor, SparseFrameAccessor
from .base import _ATTRS_NO_LOOKUP, BasePandasDataset
from .groupby import DataFrameGroupBy
from .iterator import PartitionIterator
from .series import Series
from .utils import (
def _update_var_dicts_in_kwargs(self, expr, kwargs) -> None:
    """
        Copy variables with "@" prefix in `local_dict` and `global_dict` keys of kwargs.

        Parameters
        ----------
        expr : str
            The expression string to search variables with "@" prefix.
        kwargs : dict
            See the documentation for eval() for complete details on the keyword arguments accepted by query().
        """
    if '@' not in expr:
        return
    frame = sys._getframe()
    try:
        f_locals = frame.f_back.f_back.f_back.f_back.f_locals
        f_globals = frame.f_back.f_back.f_back.f_back.f_globals
    finally:
        del frame
    local_names = set(re.findall('@([\\w]+)', expr))
    local_dict = {}
    global_dict = {}
    for name in local_names:
        for dct_out, dct_in in ((local_dict, f_locals), (global_dict, f_globals)):
            try:
                dct_out[name] = dct_in[name]
            except KeyError:
                pass
    if local_dict:
        local_dict.update(kwargs.get('local_dict') or {})
        kwargs['local_dict'] = local_dict
    if global_dict:
        global_dict.update(kwargs.get('global_dict') or {})
        kwargs['global_dict'] = global_dict