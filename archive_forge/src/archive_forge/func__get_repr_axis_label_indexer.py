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
def _get_repr_axis_label_indexer(labels, num_for_repr):
    """
    Get the indexer for the given axis labels to be used for the repr.

    Parameters
    ----------
    labels : pandas.Index
        The axis labels.
    num_for_repr : int
        The number of elements to display.

    Returns
    -------
    slice or list
        The indexer to use for the repr.
    """
    if len(labels) <= num_for_repr:
        return slice(None)
    if num_for_repr % 2 == 0:
        front_repr_num = num_for_repr // 2 + 1
        back_repr_num = num_for_repr // 2
    else:
        front_repr_num = num_for_repr // 2 + 1
        back_repr_num = num_for_repr // 2 + 1
    all_positions = range(len(labels))
    return list(all_positions[:front_repr_num]) + ([] if back_repr_num == 0 else list(all_positions[-back_repr_num:]))