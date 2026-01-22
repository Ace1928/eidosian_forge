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
def _window_lead_lag(self, l_func: Callable, df: DataFrame, func: WindowFunctionSpec, args: List[ArgumentSpec], dest_col_name: str) -> DataFrame:
    assert_or_throw(not (func.has_order_by and (not func.has_partition_by)), ValueError('for dask engine, order by requires partition by'))
    assert_or_throw(func.has_partition_by, ValueError('for dask engine, partition by is required for lead/lag'))
    assert_or_throw(len(args) in [2, 3], ValueError(f'{args}'))
    assert_or_throw(args[0].is_col, ValueError(f'{args[0]}'))
    assert_or_throw(not args[1].is_col and isinstance(args[1].value, int), ValueError(f'{args[1]}'))
    if len(args) == 3:
        assert_or_throw(not args[2].is_col, ValueError(f'{args[2]}'))
    assert_or_throw(not func.has_windowframe, ValueError(f"lead/lag functions can't have windowframe {func}"))
    col = args[0].value
    delta = int(args[1].value)
    default = None if len(args) == 2 else args[2].value
    ndf = self.to_native(df)
    keys = list(df.keys())
    ndf, sort_keys, asc = self._prepare_sort(ndf, func.window.order_by)
    gp, gp_keys, _ = self._safe_groupby(ndf, func.window.partition_keys)

    def gp_apply(gdf: Any) -> Any:
        gdf = gdf.sort_values(sort_keys, ascending=asc)
        gdf[dest_col_name] = l_func(gdf[col], delta, default)
        return gdf.reset_index(drop=True)
    meta = [(ndf[x].name, ndf[x].dtype) for x in ndf.columns]
    meta += [(dest_col_name, ndf[col].dtype)]
    ndf = gp.apply(gp_apply, meta=meta).reset_index(drop=True)
    return self.to_df(ndf[keys + [dest_col_name]])