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
@pandas.util.cache_readonly
def _is_dataframe(self) -> bool:
    """
        Tell whether this is a dataframe.

        Ideally, other methods of BasePandasDataset shouldn't care whether this
        is a dataframe or a series, but sometimes we need to know. This method
        is better than hasattr(self, "columns"), which for series will call
        self.__getattr__("columns"), which requires materializing the index.

        Returns
        -------
        bool : Whether this is a dataframe.
        """
    return issubclass(self._pandas_class, pandas.DataFrame)