import warnings
from numpy import zeros, concatenate, ravel, diff, array, ones  # noqa:F401
import numpy as np
from . import _fitpack_impl
from . import dfitpack
class SmoothSphereBivariateSpline(SphereBivariateSpline):
    """
    Smooth bivariate spline approximation in spherical coordinates.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    theta, phi, r : array_like
        1-D sequences of data points (order is not important). Coordinates
        must be given in radians. Theta must lie within the interval
        ``[0, pi]``, and phi must lie within the interval ``[0, 2pi]``.
    w : array_like, optional
        Positive 1-D sequence of weights.
    s : float, optional
        Positive smoothing factor defined for estimation condition:
        ``sum((w(i)*(r(i) - s(theta(i), phi(i))))**2, axis=0) <= s``
        Default ``s=len(w)`` which should be a good value if ``1/w[i]`` is an
        estimate of the standard deviation of ``r[i]``.
    eps : float, optional
        A threshold for determining the effective rank of an over-determined
        linear system of equations. `eps` should have a value within the open
        interval ``(0, 1)``, the default is 1e-16.

    See Also
    --------
    BivariateSpline :
        a base class for bivariate splines.
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    LSQBivariateSpline :
        a bivariate spline using weighted least-squares fitting
    RectSphereBivariateSpline :
        a bivariate spline over a rectangular mesh on a sphere
    LSQSphereBivariateSpline :
        a bivariate spline in spherical coordinates using weighted
        least-squares fitting
    RectBivariateSpline :
        a bivariate spline over a rectangular mesh.
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives

    Notes
    -----
    For more information, see the FITPACK_ site about this function.

    .. _FITPACK: http://www.netlib.org/dierckx/sphere.f

    Examples
    --------
    Suppose we have global data on a coarse grid (the input data does not
    have to be on a grid):

    >>> import numpy as np
    >>> theta = np.linspace(0., np.pi, 7)
    >>> phi = np.linspace(0., 2*np.pi, 9)
    >>> data = np.empty((theta.shape[0], phi.shape[0]))
    >>> data[:,0], data[0,:], data[-1,:] = 0., 0., 0.
    >>> data[1:-1,1], data[1:-1,-1] = 1., 1.
    >>> data[1,1:-1], data[-2,1:-1] = 1., 1.
    >>> data[2:-2,2], data[2:-2,-2] = 2., 2.
    >>> data[2,2:-2], data[-3,2:-2] = 2., 2.
    >>> data[3,3:-2] = 3.
    >>> data = np.roll(data, 4, 1)

    We need to set up the interpolator object

    >>> lats, lons = np.meshgrid(theta, phi)
    >>> from scipy.interpolate import SmoothSphereBivariateSpline
    >>> lut = SmoothSphereBivariateSpline(lats.ravel(), lons.ravel(),
    ...                                   data.T.ravel(), s=3.5)

    As a first test, we'll see what the algorithm returns when run on the
    input coordinates

    >>> data_orig = lut(theta, phi)

    Finally we interpolate the data to a finer grid

    >>> fine_lats = np.linspace(0., np.pi, 70)
    >>> fine_lons = np.linspace(0., 2 * np.pi, 90)

    >>> data_smth = lut(fine_lats, fine_lons)

    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(131)
    >>> ax1.imshow(data, interpolation='nearest')
    >>> ax2 = fig.add_subplot(132)
    >>> ax2.imshow(data_orig, interpolation='nearest')
    >>> ax3 = fig.add_subplot(133)
    >>> ax3.imshow(data_smth, interpolation='nearest')
    >>> plt.show()

    """

    def __init__(self, theta, phi, r, w=None, s=0.0, eps=1e-16):
        theta, phi, r = (np.asarray(theta), np.asarray(phi), np.asarray(r))
        if not ((0.0 <= theta).all() and (theta <= np.pi).all()):
            raise ValueError('theta should be between [0, pi]')
        if not ((0.0 <= phi).all() and (phi <= 2.0 * np.pi).all()):
            raise ValueError('phi should be between [0, 2pi]')
        if w is not None:
            w = np.asarray(w)
            if not (w >= 0.0).all():
                raise ValueError('w should be positive')
        if not s >= 0.0:
            raise ValueError('s should be positive')
        if not 0.0 < eps < 1.0:
            raise ValueError('eps should be between (0, 1)')
        nt_, tt_, np_, tp_, c, fp, ier = dfitpack.spherfit_smth(theta, phi, r, w=w, s=s, eps=eps)
        if ier not in [0, -1, -2]:
            message = _spherefit_messages.get(ier, 'ier=%s' % ier)
            raise ValueError(message)
        self.fp = fp
        self.tck = (tt_[:nt_], tp_[:np_], c[:(nt_ - 4) * (np_ - 4)])
        self.degrees = (3, 3)

    def __call__(self, theta, phi, dtheta=0, dphi=0, grid=True):
        theta = np.asarray(theta)
        phi = np.asarray(phi)
        if phi.size > 0 and (phi.min() < 0.0 or phi.max() > 2.0 * np.pi):
            raise ValueError('requested phi out of bounds.')
        return SphereBivariateSpline.__call__(self, theta, phi, dtheta=dtheta, dphi=dphi, grid=grid)