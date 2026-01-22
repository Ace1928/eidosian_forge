from typing import Optional
import warnings
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from geopandas import _compat as compat
from geopandas.array import _check_crs, _crs_mismatch_warn
def _nearest_query(left_df: GeoDataFrame, right_df: GeoDataFrame, max_distance: float, how: str, return_distance: bool, exclusive: bool):
    if not (compat.USE_SHAPELY_20 or (compat.USE_PYGEOS and compat.PYGEOS_GE_010)):
        raise NotImplementedError('Currently, only PyGEOS >= 0.10.0 or Shapely >= 2.0 supports `nearest_all`. ' + compat.INSTALL_PYGEOS_ERROR)
    use_left_as_sindex = how == 'right'
    if use_left_as_sindex:
        sindex = left_df.sindex
        query = right_df.geometry
    else:
        sindex = right_df.sindex
        query = left_df.geometry
    if sindex:
        res = sindex.nearest(query, return_all=True, max_distance=max_distance, return_distance=return_distance, exclusive=exclusive)
        if return_distance:
            (input_idx, tree_idx), distances = res
        else:
            input_idx, tree_idx = res
            distances = None
        if use_left_as_sindex:
            l_idx, r_idx = (tree_idx, input_idx)
            sort_order = np.argsort(l_idx, kind='stable')
            l_idx, r_idx = (l_idx[sort_order], r_idx[sort_order])
            if distances is not None:
                distances = distances[sort_order]
        else:
            l_idx, r_idx = (input_idx, tree_idx)
        join_df = pd.DataFrame({'_key_left': l_idx, '_key_right': r_idx, 'distances': distances})
    else:
        join_df = pd.DataFrame(columns=['_key_left', '_key_right', 'distances'], dtype=float)
    return join_df