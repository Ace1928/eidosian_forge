from __future__ import annotations
import itertools
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal, Union
import pandas as pd
from xarray.core import dtypes
from xarray.core.concat import concat
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.merge import merge
from xarray.core.utils import iterate_nested
def _check_dimension_depth_tile_ids(combined_tile_ids):
    """
    Check all tuples are the same length, i.e. check that all lists are
    nested to the same depth.
    """
    tile_ids = combined_tile_ids.keys()
    nesting_depths = [len(tile_id) for tile_id in tile_ids]
    if not nesting_depths:
        nesting_depths = [0]
    if set(nesting_depths) != {nesting_depths[0]}:
        raise ValueError('The supplied objects do not form a hypercube because sub-lists do not have consistent depths')
    return (tile_ids, nesting_depths)