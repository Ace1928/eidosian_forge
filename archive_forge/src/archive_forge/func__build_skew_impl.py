import numpy as np
import pandas
from modin.config import use_range_partitioning_groupby
from modin.core.dataframe.algebra import GroupByReduce
from modin.error_message import ErrorMessage
from modin.utils import hashable
@staticmethod
def _build_skew_impl():
    """
        Build TreeReduce implementation for 'skew' groupby aggregation.

        Returns
        -------
        (map_fn: callable, reduce_fn: callable, default2pandas_fn: callable)
        """

    def skew_map(dfgb, *args, **kwargs):
        if dfgb._selection is not None:
            data_to_agg = dfgb._selected_obj
        else:
            cols_to_agg = dfgb.obj.columns.difference(dfgb.exclusions)
            data_to_agg = dfgb.obj[cols_to_agg]
        df_pow2 = data_to_agg ** 2
        df_pow3 = data_to_agg ** 3
        return pandas.concat([dfgb.count(*args, **kwargs), dfgb.sum(*args, **kwargs), df_pow2.groupby(dfgb.grouper).sum(*args, **kwargs), df_pow3.groupby(dfgb.grouper).sum(*args, **kwargs)], copy=False, axis=1, keys=['count', 'sum', 'pow2_sum', 'pow3_sum'], names=[GroupByReduce.ID_LEVEL_NAME])

    def skew_reduce(dfgb, *args, **kwargs):
        df = dfgb.sum(*args, **kwargs)
        if df.empty:
            return df.droplevel(GroupByReduce.ID_LEVEL_NAME, axis=1)
        count = df['count']
        s = df['sum']
        s2 = df['pow2_sum']
        s3 = df['pow3_sum']
        m = s / count
        m2 = s2 - 2 * m * s + count * m ** 2
        m3 = s3 - 3 * m * s2 + 3 * s * m ** 2 - count * m ** 3
        with np.errstate(invalid='ignore', divide='ignore'):
            skew_res = count * (count - 1) ** 0.5 / (count - 2) * (m3 / m2 ** 1.5)
        skew_res[m2 == 0] = 0
        skew_res[count < 3] = np.nan
        return skew_res
    GroupByReduce.register_implementation(skew_map, skew_reduce)
    return (skew_map, skew_reduce, lambda grp, *args, **kwargs: grp.skew(*args, **kwargs))