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
def _window_ranks(self, rank_func: Callable, df: DataFrame, func: WindowFunctionSpec, args: List[ArgumentSpec], dest_col_name: str) -> DataFrame:
    assert_or_throw(len(args) == 0, ValueError(f'{args}'))
    assert_or_throw(func.has_order_by, ValueError(f'rank functions require order by {func}'))
    assert_or_throw(not func.has_windowframe, ValueError(f"rank functions can't have windowframe {func}"))
    assert_or_throw(not (func.has_order_by and (not func.has_partition_by)), ValueError('for dask engine, order by requires partition by'))
    assert_or_throw(func.has_partition_by, ValueError('for dask engine, partition by is required for ranking'))
    ndf = self.to_native(df)
    keys = list(df.keys())
    ndf, sort_keys, asc = self._prepare_sort(ndf, func.window.order_by)
    gp, gp_keys, _ = self._safe_groupby(ndf, func.window.partition_keys)

    def gp_apply(gdf: Any) -> Any:
        gdf = gdf.sort_values(sort_keys, ascending=asc)
        rank_on = self._safe_pandas_groupby(gdf, sort_keys, as_index=False, sort=False)[0].ngroup()
        gdf[dest_col_name] = rank_func(rank_on)
        return gdf.reset_index(drop=True)
    meta = [(ndf[x].name, ndf[x].dtype) for x in ndf.columns]
    meta += [(dest_col_name, 'f8')]
    ndf = gp.apply(gp_apply, meta=meta).reset_index(drop=True)
    return self.to_df(ndf[keys + [dest_col_name]])