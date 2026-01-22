import numpy as np
from ._util import _validate_connectivity, _offsets_to_raveled_neighbors
from ..util import invert
from . import _max_tree
def area_closing(image, area_threshold=64, connectivity=1, parent=None, tree_traverser=None):
    """Perform an area closing of the image.

    Area closing removes all dark structures of an image with
    a surface smaller than area_threshold.
    The output image is larger than or equal to the input image
    for every pixel and all local minima have at least a surface of
    area_threshold pixels.

    Area closings are similar to morphological closings, but
    they do not use a fixed footprint, but rather a deformable
    one, with surface = area_threshold.

    In the binary case, area closings are equivalent to
    remove_small_holes; this operator is thus extended to gray-level images.

    Technically, this operator is based on the max-tree representation of
    the image.

    Parameters
    ----------
    image : ndarray
        The input image for which the area_closing is to be calculated.
        This image can be of any type.
    area_threshold : unsigned int
        The size parameter (number of pixels). The default value is arbitrarily
        chosen to be 64.
    connectivity : unsigned int, optional
        The neighborhood connectivity. The integer represents the maximum
        number of orthogonal steps to reach a neighbor. In 2D, it is 1 for
        a 4-neighborhood and 2 for a 8-neighborhood. Default value is 1.
    parent : ndarray, int64, optional
        Parent image representing the max tree of the inverted image. The
        value of each pixel is the index of its parent in the ravelled array.
        See Note for further details.
    tree_traverser : 1D array, int64, optional
        The ordered pixel indices (referring to the ravelled array). The pixels
        are ordered such that every pixel is preceded by its parent (except for
        the root which has no parent).

    Returns
    -------
    output : ndarray
        Output image of the same shape and type as input image.

    See Also
    --------
    skimage.morphology.area_opening
    skimage.morphology.diameter_opening
    skimage.morphology.diameter_closing
    skimage.morphology.max_tree
    skimage.morphology.remove_small_objects
    skimage.morphology.remove_small_holes

    References
    ----------
    .. [1] Vincent L., Proc. "Grayscale area openings and closings,
           their efficient implementation and applications",
           EURASIP Workshop on Mathematical Morphology and its
           Applications to Signal Processing, Barcelona, Spain, pp.22-27,
           May 1993.
    .. [2] Soille, P., "Morphological Image Analysis: Principles and
           Applications" (Chapter 6), 2nd edition (2003), ISBN 3540429883.
           :DOI:`10.1007/978-3-662-05088-0`
    .. [3] Salembier, P., Oliveras, A., & Garrido, L. (1998). Antiextensive
           Connected Operators for Image and Sequence Processing.
           IEEE Transactions on Image Processing, 7(4), 555-570.
           :DOI:`10.1109/83.663500`
    .. [4] Najman, L., & Couprie, M. (2006). Building the component tree in
           quasi-linear time. IEEE Transactions on Image Processing, 15(11),
           3531-3539.
           :DOI:`10.1109/TIP.2006.877518`
    .. [5] Carlinet, E., & Geraud, T. (2014). A Comparative Review of
           Component Tree Computation Algorithms. IEEE Transactions on Image
           Processing, 23(9), 3885-3895.
           :DOI:`10.1109/TIP.2014.2336551`

    Examples
    --------
    We create an image (quadratic function with a minimum in the center and
    4 additional local minima.

    >>> w = 12
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 180 + 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:3,1:5] = 160; f[2:4,9:11] = 140; f[9:11,2:4] = 120
    >>> f[9:10,9:11] = 100; f[10,10] = 100
    >>> f = f.astype(int)

    We can calculate the area closing:

    >>> closed = area_closing(f, 8, connectivity=1)

    All small minima are removed, and the remaining minima have at least
    a size of 8.

    Notes
    -----
    If a max-tree representation (parent and tree_traverser) are given to the
    function, they must be calculated from the inverted image for this
    function, i.e.:
    >>> P, S = max_tree(invert(f))
    >>> closed = diameter_closing(f, 3, parent=P, tree_traverser=S)
    """
    image_inv = invert(image)
    output = image_inv.copy()
    if parent is None or tree_traverser is None:
        parent, tree_traverser = max_tree(image_inv, connectivity)
    area = _max_tree._compute_area(image_inv.ravel(), parent.ravel(), tree_traverser)
    _max_tree._direct_filter(image_inv.ravel(), output.ravel(), parent.ravel(), tree_traverser, area, area_threshold)
    output = invert(output)
    return output