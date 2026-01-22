from enum import Enum
from typing import TYPE_CHECKING, Callable, Tuple
import numpy as np
import pandas
from pandas.core.dtypes.common import is_numeric_dtype
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
@staticmethod
def _maybe_combine_nan_and_non_nan_aggs(total_agg: pandas.DataFrame) -> pandas.DataFrame:
    """
        Pair the aggregation results of partitions having and not having NaN values if needed.

        Parameters
        ----------
        total_agg : pandas.DataFrame
            A dataframe holding aggregations computed for each partition
            concatenated along the rows axis.

        Returns
        -------
        pandas.DataFrame
            DataFrame with aligned results.
        """
    nsums = total_agg.columns.get_locs(['sum'])
    if not (len(nsums) > 1 and ('sum', MODIN_UNNAMED_SERIES_LABEL) in total_agg.columns):
        return total_agg
    cols = total_agg.columns
    all_agg_idxs = np.where(cols.get_loc('sum') | cols.get_loc('pow2_sum') | cols.get_loc('count'))[0]
    non_na_agg_idxs = cols.get_indexer_for(pandas.Index([('sum', MODIN_UNNAMED_SERIES_LABEL), ('pow2_sum', MODIN_UNNAMED_SERIES_LABEL), ('count', MODIN_UNNAMED_SERIES_LABEL)]))
    na_agg_idxs = np.setdiff1d(all_agg_idxs, non_na_agg_idxs, assume_unique=True)
    parts_with_nans = total_agg.values[:, na_agg_idxs]
    parts_without_nans = total_agg.values[:, non_na_agg_idxs].repeat(repeats=len(parts_with_nans), axis=0).reshape(parts_with_nans.shape, order='F')
    replace_values = parts_with_nans + parts_without_nans
    if not total_agg.values.flags.writeable:
        total_agg = total_agg.copy()
    total_agg.values[:, na_agg_idxs] = replace_values
    return total_agg