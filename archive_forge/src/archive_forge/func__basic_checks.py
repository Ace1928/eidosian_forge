from typing import Optional
import warnings
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from geopandas import _compat as compat
from geopandas.array import _check_crs, _crs_mismatch_warn
def _basic_checks(left_df, right_df, how, lsuffix, rsuffix):
    """Checks the validity of join input parameters.

    `how` must be one of the valid options.
    `'index_'` concatenated with `lsuffix` or `rsuffix` must not already
    exist as columns in the left or right data frames.

    Parameters
    ------------
    left_df : GeoDataFrame
    right_df : GeoData Frame
    how : str, one of 'left', 'right', 'inner'
        join type
    lsuffix : str
        left index suffix
    rsuffix : str
        right index suffix
    """
    if not isinstance(left_df, GeoDataFrame):
        raise ValueError("'left_df' should be GeoDataFrame, got {}".format(type(left_df)))
    if not isinstance(right_df, GeoDataFrame):
        raise ValueError("'right_df' should be GeoDataFrame, got {}".format(type(right_df)))
    allowed_hows = ['left', 'right', 'inner']
    if how not in allowed_hows:
        raise ValueError('`how` was "{}" but is expected to be in {}'.format(how, allowed_hows))
    if not _check_crs(left_df, right_df):
        _crs_mismatch_warn(left_df, right_df, stacklevel=4)
    index_left = 'index_{}'.format(lsuffix)
    index_right = 'index_{}'.format(rsuffix)
    if any(left_df.columns.isin([index_left, index_right])) or any(right_df.columns.isin([index_left, index_right])):
        raise ValueError("'{0}' and '{1}' cannot be names in the frames being joined".format(index_left, index_right))