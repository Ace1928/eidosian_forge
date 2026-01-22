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
def _group_agg_no_keys(self, df: DataFrame, agg_map: Dict[str, Tuple[str, AggFunctionSpec]]) -> DataFrame:
    ndf = self.to_native(df)
    aggs: Dict[str, Any] = {}
    for k, v in agg_map.items():
        if '*' in v[0] or ',' in v[0]:
            continue
        aggs[k] = self._agg_no_gp(v[1], ndf[v[0]])
    for k, v in agg_map.items():
        if '*' in v[0] or ',' in v[0]:
            if v[1].unique:
                aggs[k] = self._count_unique_no_group(ndf, v[0], v[1])
            else:
                aggs[k] = self._count_all_no_group(ndf, v[1])
    data: Dict[str, Any] = {}
    for k, vv in aggs.items():
        s = vv.compute() if hasattr(vv, 'compute') else vv
        if isinstance(s, pandas.Series):
            data[k] = [s.iloc[0]]
        else:
            data[k] = [s]
    return self.to_df(pandas.DataFrame(data))