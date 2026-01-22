import numpy as np
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type
from ..color import rgb2lab
from ..util import img_as_float
from ._quickshift_cy import _quickshift_cython
Segment image using quickshift clustering in Color-(x,y) space.

    Produces an oversegmentation of the image using the quickshift mode-seeking
    algorithm.

    Parameters
    ----------
    image : (M, N, C) ndarray
        Input image. The axis corresponding to color channels can be specified
        via the `channel_axis` argument.
    ratio : float, optional, between 0 and 1
        Balances color-space proximity and image-space proximity.
        Higher values give more weight to color-space.
    kernel_size : float, optional
        Width of Gaussian kernel used in smoothing the
        sample density. Higher means fewer clusters.
    max_dist : float, optional
        Cut-off point for data distances.
        Higher means fewer clusters.
    return_tree : bool, optional
        Whether to return the full segmentation hierarchy tree and distances.
    sigma : float, optional
        Width for Gaussian smoothing as preprocessing. Zero means no smoothing.
    convert2lab : bool, optional
        Whether the input should be converted to Lab colorspace prior to
        segmentation. For this purpose, the input is assumed to be RGB.
    rng : {`numpy.random.Generator`, int}, optional
        Pseudo-random number generator.
        By default, a PCG64 generator is used (see :func:`numpy.random.default_rng`).
        If `rng` is an int, it is used to seed the generator.

        The PRNG is used to break ties, and is seeded with 42 by default.
    channel_axis : int, optional
        The axis of `image` corresponding to color channels. Defaults to the
        last axis.

    Returns
    -------
    segment_mask : (M, N) ndarray
        Integer mask indicating segment labels.

    Notes
    -----
    The authors advocate to convert the image to Lab color space prior to
    segmentation, though this is not strictly necessary. For this to work, the
    image must be given in RGB format.

    References
    ----------
    .. [1] Quick shift and kernel methods for mode seeking,
           Vedaldi, A. and Soatto, S.
           European Conference on Computer Vision, 2008
    