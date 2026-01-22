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
def _agg_dtype(agg, dtype):
    """
    Compute aggregate data type.

    Parameters
    ----------
    agg : str
        Aggregate name.
    dtype : dtype
        Operand data type.

    Returns
    -------
    dtype
        The aggregate data type.
    """
    if agg in _aggs_preserving_numeric_type:
        return dtype
    elif agg in _aggs_with_int_result:
        return _get_dtype(int)
    elif agg in _aggs_with_float_result:
        return _get_dtype(float)
    elif agg == 'quantile':
        return _quantile_agg_dtype(dtype)
    else:
        raise NotImplementedError(f'unsupported aggregate {agg}')