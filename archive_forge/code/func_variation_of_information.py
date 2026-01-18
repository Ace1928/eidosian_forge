import numpy as np
import scipy.sparse as sparse
from ._contingency_table import contingency_table
from .._shared.utils import check_shape_equality
def variation_of_information(image0=None, image1=None, *, table=None, ignore_labels=()):
    """Return symmetric conditional entropies associated with the VI. [1]_

    The variation of information is defined as VI(X,Y) = H(X|Y) + H(Y|X).
    If X is the ground-truth segmentation, then H(X|Y) can be interpreted
    as the amount of under-segmentation and H(Y|X) as the amount
    of over-segmentation. In other words, a perfect over-segmentation
    will have H(X|Y)=0 and a perfect under-segmentation will have H(Y|X)=0.

    Parameters
    ----------
    image0, image1 : ndarray of int
        Label images / segmentations, must have same shape.
    table : scipy.sparse array in csr format, optional
        A contingency table built with skimage.evaluate.contingency_table.
        If None, it will be computed with skimage.evaluate.contingency_table.
        If given, the entropies will be computed from this table and any images
        will be ignored.
    ignore_labels : sequence of int, optional
        Labels to ignore. Any part of the true image labeled with any of these
        values will not be counted in the score.

    Returns
    -------
    vi : ndarray of float, shape (2,)
        The conditional entropies of image1|image0 and image0|image1.

    References
    ----------
    .. [1] Marina Meilă (2007), Comparing clusterings—an information based
        distance, Journal of Multivariate Analysis, Volume 98, Issue 5,
        Pages 873-895, ISSN 0047-259X, :DOI:`10.1016/j.jmva.2006.11.013`.
    """
    h0g1, h1g0 = _vi_tables(image0, image1, table=table, ignore_labels=ignore_labels)
    return np.array([h1g0.sum(), h0g1.sum()])