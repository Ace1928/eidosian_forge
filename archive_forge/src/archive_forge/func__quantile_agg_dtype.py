import abc
from typing import Generator, Type, Union
import numpy as np
import pandas
import pyarrow as pa
import pyarrow.compute as pc
from pandas.core.dtypes.common import (
from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .dataframe.utils import ColNameCodec, to_arrow_type
def _quantile_agg_dtype(dtype):
    """
    Compute the quantile aggregate data type.

    Parameters
    ----------
    dtype : dtype

    Returns
    -------
    dtype
    """
    return dtype if is_datetime64_any_dtype(dtype) else _get_dtype(float)