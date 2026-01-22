import numpy as np
import pandas
from modin.config import use_range_partitioning_groupby
from modin.core.dataframe.algebra import GroupByReduce
from modin.error_message import ErrorMessage
from modin.utils import hashable
@staticmethod
def _build_mean_impl():
    """
        Build TreeReduce implementation for 'mean' groupby aggregation.

        Returns
        -------
        (map_fn: callable, reduce_fn: callable, default2pandas_fn: callable)
        """

    def mean_map(dfgb, **kwargs):
        return pandas.concat([dfgb.sum(**kwargs), dfgb.count()], axis=1, copy=False, keys=['sum', 'count'], names=[GroupByReduce.ID_LEVEL_NAME])

    def mean_reduce(dfgb, **kwargs):
        """
            Compute mean value in each group using sums/counts values within reduce phase.

            Parameters
            ----------
            dfgb : pandas.DataFrameGroupBy
                GroupBy object for column-partition.
            **kwargs : dict
                Additional keyword parameters to be passed in ``pandas.DataFrameGroupBy.sum``.

            Returns
            -------
            pandas.DataFrame
                A pandas Dataframe with mean values in each column of each group.
            """
        sums_counts_df = dfgb.sum(**kwargs)
        if sums_counts_df.empty:
            return sums_counts_df.droplevel(GroupByReduce.ID_LEVEL_NAME, axis=1)
        sum_df = sums_counts_df['sum']
        count_df = sums_counts_df['count']
        return sum_df / count_df
    GroupByReduce.register_implementation(mean_map, mean_reduce)
    return (mean_map, mean_reduce, lambda grp, *args, **kwargs: grp.mean(*args, **kwargs))