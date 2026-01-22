import numpy as np
from scipy.interpolate import RectBivariateSpline
from .._shared.utils import _supported_float_type
from ..util import img_as_float
from ..filters import sobel
Active contour model.

    Active contours by fitting snakes to features of images. Supports single
    and multichannel 2D images. Snakes can be periodic (for segmentation) or
    have fixed and/or free ends.
    The output snake has the same length as the input boundary.
    As the number of points is constant, make sure that the initial snake
    has enough points to capture the details of the final contour.

    Parameters
    ----------
    image : (M, N) or (M, N, 3) ndarray
        Input image.
    snake : (K, 2) ndarray
        Initial snake coordinates. For periodic boundary conditions, endpoints
        must not be duplicated.
    alpha : float, optional
        Snake length shape parameter. Higher values makes snake contract
        faster.
    beta : float, optional
        Snake smoothness shape parameter. Higher values makes snake smoother.
    w_line : float, optional
        Controls attraction to brightness. Use negative values to attract
        toward dark regions.
    w_edge : float, optional
        Controls attraction to edges. Use negative values to repel snake from
        edges.
    gamma : float, optional
        Explicit time stepping parameter.
    max_px_move : float, optional
        Maximum pixel distance to move per iteration.
    max_num_iter : int, optional
        Maximum iterations to optimize snake shape.
    convergence : float, optional
        Convergence criteria.
    boundary_condition : string, optional
        Boundary conditions for the contour. Can be one of 'periodic',
        'free', 'fixed', 'free-fixed', or 'fixed-free'. 'periodic' attaches
        the two ends of the snake, 'fixed' holds the end-points in place,
        and 'free' allows free movement of the ends. 'fixed' and 'free' can
        be combined by parsing 'fixed-free', 'free-fixed'. Parsing
        'fixed-fixed' or 'free-free' yields same behaviour as 'fixed' and
        'free', respectively.

    Returns
    -------
    snake : (K, 2) ndarray
        Optimised snake, same shape as input parameter.

    References
    ----------
    .. [1]  Kass, M.; Witkin, A.; Terzopoulos, D. "Snakes: Active contour
            models". International Journal of Computer Vision 1 (4): 321
            (1988). :DOI:`10.1007/BF00133570`

    Examples
    --------
    >>> from skimage.draw import circle_perimeter
    >>> from skimage.filters import gaussian

    Create and smooth image:

    >>> img = np.zeros((100, 100))
    >>> rr, cc = circle_perimeter(35, 45, 25)
    >>> img[rr, cc] = 1
    >>> img = gaussian(img, sigma=2, preserve_range=False)

    Initialize spline:

    >>> s = np.linspace(0, 2*np.pi, 100)
    >>> init = 50 * np.array([np.sin(s), np.cos(s)]).T + 50

    Fit spline to image:

    >>> snake = active_contour(img, init, w_edge=0, w_line=1)  # doctest: +SKIP
    >>> dist = np.sqrt((45-snake[:, 0])**2 + (35-snake[:, 1])**2)  # doctest: +SKIP
    >>> int(np.mean(dist))  # doctest: +SKIP
    25

    