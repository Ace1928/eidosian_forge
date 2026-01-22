import numpy as np
import pandas
from modin.config import use_range_partitioning_groupby
from modin.core.dataframe.algebra import GroupByReduce
from modin.error_message import ErrorMessage
from modin.utils import hashable
def applyier(df, other):
    """
            Build pivot table for a single partition.

            Parameters
            ----------
            df : pandas.DataFrame
                Partition of the self frame.
            other : pandas.DataFrame
                Broadcasted partition that contains `value` columns
                of the self frame.

            Returns
            -------
            pandas.DataFrame
                Pivot table for this particular partition.
            """
    concated = pandas.concat([df, other], axis=1, copy=False)
    del df, other
    result = pandas.pivot_table(concated, **pivot_kwargs)
    del concated
    if drop_column_level is not None:
        result = result.droplevel(drop_column_level, axis=1)
    if len(index) == 0 and len(columns) > 0:
        result = result.transpose()
    return result