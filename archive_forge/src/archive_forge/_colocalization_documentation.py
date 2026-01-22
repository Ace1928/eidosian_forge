import numpy as np
from scipy.stats import pearsonr
from .._shared.utils import check_shape_equality, as_binary_ndarray
Fraction of a channel's segmented binary mask that overlaps with a
    second channel's segmented binary mask.

    Parameters
    ----------
    image0_mask : (M, N) ndarray of dtype bool
        Image mask of channel A.
    image1_mask : (M, N) ndarray of dtype bool
        Image mask of channel B.
        Must have same dimensions as `image0_mask`.
    mask : (M, N) ndarray of dtype bool, optional
        Only `image0_mask` and `image1_mask` pixels within this region of
        interest
        mask are included in the calculation.
        Must have same dimensions as `image0_mask`.

    Returns
    -------
    Intersection coefficient, float
        Fraction of `image0_mask` that overlaps with `image1_mask`.

    