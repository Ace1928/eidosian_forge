from typing import Optional
import warnings
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from geopandas import _compat as compat
from geopandas.array import _check_crs, _crs_mismatch_warn
def _frame_join(join_df, left_df, right_df, how, lsuffix, rsuffix):
    """Join the GeoDataFrames at the DataFrame level.

    Parameters
    ----------
    join_df : DataFrame
        Indices and join data returned by the geometric join.
        Must have columns `_key_left` and `_key_right`
        with integer indices representing the matches
        from `left_df` and `right_df` respectively.
        Additional columns may be included and will be copied to
        the resultant GeoDataFrame.
    left_df : GeoDataFrame
    right_df : GeoDataFrame
    lsuffix : string
        Suffix to apply to overlapping column names (left GeoDataFrame).
    rsuffix : string
        Suffix to apply to overlapping column names (right GeoDataFrame).
    how : string
        The type of join to use on the DataFrame level.

    Returns
    -------
    GeoDataFrame
        Joined GeoDataFrame.
    """
    index_left = 'index_{}'.format(lsuffix)
    left_df = left_df.copy(deep=True)
    try:
        left_index_name = left_df.index.name
        left_df.index = left_df.index.rename(index_left)
    except TypeError:
        index_left = ['index_{}'.format(lsuffix + str(pos)) for pos, ix in enumerate(left_df.index.names)]
        left_index_name = left_df.index.names
        left_df.index = left_df.index.rename(index_left)
    left_df = left_df.reset_index()
    index_right = 'index_{}'.format(rsuffix)
    right_df = right_df.copy(deep=True)
    try:
        right_index_name = right_df.index.name
        right_df.index = right_df.index.rename(index_right)
    except TypeError:
        index_right = ['index_{}'.format(rsuffix + str(pos)) for pos, ix in enumerate(right_df.index.names)]
        right_index_name = right_df.index.names
        right_df.index = right_df.index.rename(index_right)
    right_df = right_df.reset_index()
    if how == 'inner':
        join_df = join_df.set_index('_key_left')
        joined = left_df.merge(join_df, left_index=True, right_index=True).merge(right_df.drop(right_df.geometry.name, axis=1), left_on='_key_right', right_index=True, suffixes=('_{}'.format(lsuffix), '_{}'.format(rsuffix))).set_index(index_left).drop(['_key_right'], axis=1)
        if isinstance(index_left, list):
            joined.index.names = left_index_name
        else:
            joined.index.name = left_index_name
    elif how == 'left':
        join_df = join_df.set_index('_key_left')
        joined = left_df.merge(join_df, left_index=True, right_index=True, how='left').merge(right_df.drop(right_df.geometry.name, axis=1), how='left', left_on='_key_right', right_index=True, suffixes=('_{}'.format(lsuffix), '_{}'.format(rsuffix))).set_index(index_left).drop(['_key_right'], axis=1)
        if isinstance(index_left, list):
            joined.index.names = left_index_name
        else:
            joined.index.name = left_index_name
    else:
        joined = left_df.drop(left_df.geometry.name, axis=1).merge(join_df.merge(right_df, left_on='_key_right', right_index=True, how='right'), left_index=True, right_on='_key_left', how='right', suffixes=('_{}'.format(lsuffix), '_{}'.format(rsuffix))).set_index(index_right).drop(['_key_left', '_key_right'], axis=1).set_geometry(right_df.geometry.name)
        if isinstance(index_right, list):
            joined.index.names = right_index_name
        else:
            joined.index.name = right_index_name
    return joined