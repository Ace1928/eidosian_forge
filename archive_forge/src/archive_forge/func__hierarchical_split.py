from statsmodels.compat.python import lrange, lzip
from itertools import product
import numpy as np
from numpy import array, cumsum, iterable, r_
from pandas import DataFrame
from statsmodels.graphics import utils
def _hierarchical_split(count_dict, horizontal=True, gap=0.05):
    """
    Split a square in a hierarchical way given a contingency table.

    Hierarchically split the unit square in alternate directions
    in proportion to the subdivision contained in the contingency table
    count_dict.  This is the function that actually perform the tiling
    for the creation of the mosaic plot.  If the gap array has been specified
    it will insert a corresponding amount of space (proportional to the
    unit length), while retaining the proportionality of the tiles.

    Parameters
    ----------
    count_dict : dict
        Dictionary containing the contingency table.
        Each category should contain a non-negative number
        with a tuple as index.  It expects that all the combination
        of keys to be represents; if that is not true, will
        automatically consider the missing values as 0
    horizontal : bool
        The starting direction of the split (by default along
        the horizontal axis)
    gap : float or array of floats
        The list of gaps to be applied on each subdivision.
        If the length of the given array is less of the number
        of subcategories (or if it's a single number) it will extend
        it with exponentially decreasing gaps

    Returns
    -------
    base_rect : dict
        A dictionary containing the result of the split.
        To each key is associated a 4-tuple of coordinates
        that are required to create the corresponding rectangle:

            0 - x position of the lower left corner
            1 - y position of the lower left corner
            2 - width of the rectangle
            3 - height of the rectangle
    """
    base_rect = dict([(tuple(), (0, 0, 1, 1))])
    categories_levels = _categories_level(list(count_dict.keys()))
    L = len(categories_levels)
    if not np.iterable(gap):
        gap = [gap / 1.5 ** idx for idx in range(L)]
    if len(gap) < L:
        last = gap[-1]
        gap = list(*gap) + [last / 1.5 ** idx for idx in range(L)]
    gap = gap[:L]
    count_ordered = {k: count_dict[k] for k in list(product(*categories_levels))}
    for cat_idx, cat_enum in enumerate(categories_levels):
        base_keys = list(product(*categories_levels[:cat_idx]))
        for key in base_keys:
            part_count = [_reduce_dict(count_ordered, key + (partial,)) for partial in cat_enum]
            new_gap = gap[cat_idx]
            base_rect = _key_splitting(base_rect, cat_enum, part_count, key, horizontal, new_gap)
        horizontal = not horizontal
    return base_rect