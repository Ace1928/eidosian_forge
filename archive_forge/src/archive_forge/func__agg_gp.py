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
def _agg_gp(self, func: AggFunctionSpec, gp: Any, dtype: Any, index: pandas.MultiIndex) -> Tuple[Any, bool]:
    name = func.name.lower()
    if name == 'sum':
        if not func.unique:
            return (gp.sum(min_count=1), True)
        else:
            return (gp.apply(lambda x: x.drop_duplicates().sum(min_count=1), meta=pandas.Series(dtype=dtype, index=index)), False)
    if name in ['avg', 'mean']:
        if not func.unique:
            return (gp.mean(), True)
        else:
            return (gp.apply(lambda x: x.drop_duplicates().mean(), meta=pandas.Series(dtype='f8', index=index)), False)
    if name in ['first', 'first_value']:
        if func.dropna:
            return (gp.first(), True)
        else:
            return (gp.first(), True)
    if name in ['last', 'last_value']:
        if func.dropna:
            return (gp.last(), True)
        else:
            return (gp.last(), True)
    if name == 'count':
        if not func.unique and (not func.dropna):
            return (gp.size(), True)
        if func.unique:
            if func.dropna:
                return (gp.nunique(), True)
            else:
                return (gp.apply(lambda x: x.drop_duplicates().size, meta=pandas.Series(dtype='i8', index=index)), False)
        if not func.unique and func.dropna:
            return (gp.count(), True)
    if name == 'min':
        if numpy.issubdtype(dtype, numpy.number):
            return (gp.min(), True)

        def min_(s: Any) -> Any:
            return s.apply(lambda x: x.dropna().min())
        agg = pd.Aggregation('min', min_, min_)
        return (gp.agg(agg), True)
    if name == 'max':
        if numpy.issubdtype(dtype, numpy.number):
            return (gp.max(), True)

        def max_(s: Any) -> Any:
            return s.apply(lambda x: x.dropna().max())
        agg = pd.Aggregation('max', max_, max_)
        return (gp.agg(agg), True)
    raise NotImplementedError