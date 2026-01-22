from typing import Any, Callable, Dict, List, Tuple
import dask.array as np
import dask.dataframe as pd
import numpy
import pandas
from triad.utils.assertion import assert_or_throw
from triad.utils.pandas_like import PandasLikeUtils
from qpd import QPDEngine, run_sql
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_pandas.engine import _RowsIndexer
def _count_all_no_group(self, df: Any, func: AggFunctionSpec) -> Any:
    assert_or_throw('count' == func.name.lower() and (not func.unique), ValueError(f'{func} is invalid'))
    return df.shape[0]