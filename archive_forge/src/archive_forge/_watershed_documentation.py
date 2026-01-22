import numpy as np
from scipy import ndimage as ndi
from . import _watershed_cy
from ..morphology.extrema import local_minima
from ..morphology._util import _validate_connectivity, _offsets_to_raveled_neighbors
from ..util import crop, regular_seeds
Find watershed basins in an image flooded from given markers.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Data array where the lowest value points are labeled first.
    markers : int, or (M, N[, ...]) ndarray of int, optional
        The desired number of basins, or an array marking the basins with the
        values to be assigned in the label matrix. Zero means not a marker. If
        None, the (default) markers are determined as the local minima of
        `image`. Specifically, the computation is equivalent to applying
        :func:`skimage.morphology.local_minima` onto `image`, followed by
        :func:`skimage.measure.label` onto the result (with the same given
        `connectivity`). Generally speaking, users are encouraged to pass
        markers explicitly.
    connectivity : ndarray, optional
        An array with the same number of dimensions as `image` whose
        non-zero elements indicate neighbors for connection.
        Following the scipy convention, default is a one-connected array of
        the dimension of the image.
    offset : array_like of shape image.ndim, optional
        offset of the connectivity (one offset per dimension)
    mask : (M, N[, ...]) ndarray of bools or 0's and 1's, optional
        Array of same shape as `image`. Only points at which mask == True
        will be labeled.
    compactness : float, optional
        Use compact watershed [1]_ with given compactness parameter.
        Higher values result in more regularly-shaped watershed basins.
    watershed_line : bool, optional
        If True, a one-pixel wide line separates the regions
        obtained by the watershed algorithm. The line has the label 0.
        Note that the method used for adding this line expects that
        marker regions are not adjacent; the watershed line may not catch
        borders between adjacent marker regions.

    Returns
    -------
    out : ndarray
        A labeled matrix of the same type and shape as `markers`.

    See Also
    --------
    skimage.segmentation.random_walker
        A segmentation algorithm based on anisotropic diffusion, usually
        slower than the watershed but with good results on noisy data and
        boundaries with holes.

    Notes
    -----
    This function implements a watershed algorithm [2]_ [3]_ that apportions
    pixels into marked basins. The algorithm uses a priority queue to hold
    the pixels with the metric for the priority queue being pixel value, then
    the time of entry into the queue -- this settles ties in favor of the
    closest marker.

    Some ideas are taken from [4]_.
    The most important insight in the paper is that entry time onto the queue
    solves two problems: a pixel should be assigned to the neighbor with the
    largest gradient or, if there is no gradient, pixels on a plateau should
    be split between markers on opposite sides.

    This implementation converts all arguments to specific, lowest common
    denominator types, then passes these to a C algorithm.

    Markers can be determined manually, or automatically using for example
    the local minima of the gradient of the image, or the local maxima of the
    distance function to the background for separating overlapping objects
    (see example).

    References
    ----------
    .. [1] P. Neubert and P. Protzel, "Compact Watershed and Preemptive SLIC:
           On Improving Trade-offs of Superpixel Segmentation Algorithms,"
           2014 22nd International Conference on Pattern Recognition,
           Stockholm, Sweden, 2014, pp. 996-1001, :DOI:`10.1109/ICPR.2014.181`
           https://www.tu-chemnitz.de/etit/proaut/publications/cws_pSLIC_ICPR.pdf

    .. [2] https://en.wikipedia.org/wiki/Watershed_%28image_processing%29

    .. [3] http://cmm.ensmp.fr/~beucher/wtshed.html

    .. [4] P. J. Soille and M. M. Ansoult, "Automated basin delineation from
           digital elevation models using mathematical morphology," Signal
           Processing, 20(2):171-182, :DOI:`10.1016/0165-1684(90)90127-K`

    Examples
    --------
    The watershed algorithm is useful to separate overlapping objects.

    We first generate an initial image with two overlapping circles:

    >>> x, y = np.indices((80, 80))
    >>> x1, y1, x2, y2 = 28, 28, 44, 52
    >>> r1, r2 = 16, 20
    >>> mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
    >>> mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
    >>> image = np.logical_or(mask_circle1, mask_circle2)

    Next, we want to separate the two circles. We generate markers at the
    maxima of the distance to the background:

    >>> from scipy import ndimage as ndi
    >>> distance = ndi.distance_transform_edt(image)
    >>> from skimage.feature import peak_local_max
    >>> max_coords = peak_local_max(distance, labels=image,
    ...                             footprint=np.ones((3, 3)))
    >>> local_maxima = np.zeros_like(image, dtype=bool)
    >>> local_maxima[tuple(max_coords.T)] = True
    >>> markers = ndi.label(local_maxima)[0]

    Finally, we run the watershed on the image and markers:

    >>> labels = watershed(-distance, markers, mask=image)

    The algorithm works also for 3-D images, and can be used for example to
    separate overlapping spheres.
    