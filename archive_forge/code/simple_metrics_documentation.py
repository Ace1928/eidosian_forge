import numpy as np
from scipy.stats import entropy
from ..util.dtype import dtype_range
from .._shared.utils import _supported_float_type, check_shape_equality, warn
Compute the normalized mutual information (NMI).

    The normalized mutual information of :math:`A` and :math:`B` is given by::

    .. math::

        Y(A, B) = \frac{H(A) + H(B)}{H(A, B)}

    where :math:`H(X) := - \sum_{x \in X}{x \log x}` is the entropy.

    It was proposed to be useful in registering images by Colin Studholme and
    colleagues [1]_. It ranges from 1 (perfectly uncorrelated image values)
    to 2 (perfectly correlated image values, whether positively or negatively).

    Parameters
    ----------
    image0, image1 : ndarray
        Images to be compared. The two input images must have the same number
        of dimensions.
    bins : int or sequence of int, optional
        The number of bins along each axis of the joint histogram.

    Returns
    -------
    nmi : float
        The normalized mutual information between the two arrays, computed at
        the granularity given by ``bins``. Higher NMI implies more similar
        input images.

    Raises
    ------
    ValueError
        If the images don't have the same number of dimensions.

    Notes
    -----
    If the two input images are not the same shape, the smaller image is padded
    with zeros.

    References
    ----------
    .. [1] C. Studholme, D.L.G. Hill, & D.J. Hawkes (1999). An overlap
           invariant entropy measure of 3D medical image alignment.
           Pattern Recognition 32(1):71-86
           :DOI:`10.1016/S0031-3203(98)00091-0`
    