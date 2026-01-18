import abc
import warnings
from typing import Hashable, List, Optional
import numpy as np
import pandas
import pandas.core.resample
from pandas._typing import DtypeBackend, IndexLabel, Suffixes
from pandas.core.dtypes.common import is_number, is_scalar
from modin.config import StorageFormat
from modin.core.dataframe.algebra.default2pandas import (
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, try_cast_to_pandas
from . import doc_utils
@doc_utils.doc_window_method(window_cls_name='Rolling', win_type='window of the specified type', result='variance', refer_to='var', params='\n        ddof : int, default: 1\n        *args : iterable\n        **kwargs : dict')
def window_var(self, fold_axis, window_kwargs, ddof=1, *args, **kwargs):
    return RollingDefault.register(pandas.core.window.Window.var)(self, window_kwargs, ddof, *args, **kwargs)