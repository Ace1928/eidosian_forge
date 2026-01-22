import warnings
from numpy import zeros, concatenate, ravel, diff, array, ones  # noqa:F401
import numpy as np
from . import _fitpack_impl
from . import dfitpack
class LSQSphereBivariateSpline(SphereBivariateSpline):
    """
    Weighted least-squares bivariate spline approximation in spherical
    coordinates.

    Determines a smoothing bicubic spline according to a given
    set of knots in the `theta` and `phi` directions.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    theta, phi, r : array_like
        1-D sequences of data points (order is not important). Coordinates
        must be given in radians. Theta must lie within the interval
        ``[0, pi]``, and phi must lie within the interval ``[0, 2pi]``.
    tt, tp : array_like
        Strictly ordered 1-D sequences of knots coordinates.
        Coordinates must satisfy ``0 < tt[i] < pi``, ``0 < tp[i] < 2*pi``.
    w : array_like, optional
        Positive 1-D sequence of weights, of the same length as `theta`, `phi`
        and `r`.
    eps : float, optional
        A threshold for determining the effective rank of an over-determined
        linear system of equations. `eps` should have a value within the
        open interval ``(0, 1)``, the default is 1e-16.

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
    SmoothSphereBivariateSpline :
        a smoothing bivariate spline in spherical coordinates
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

    >>> from scipy.interpolate import LSQSphereBivariateSpline
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> theta = np.linspace(0, np.pi, num=7)
    >>> phi = np.linspace(0, 2*np.pi, num=9)
    >>> data = np.empty((theta.shape[0], phi.shape[0]))
    >>> data[:,0], data[0,:], data[-1,:] = 0., 0., 0.
    >>> data[1:-1,1], data[1:-1,-1] = 1., 1.
    >>> data[1,1:-1], data[-2,1:-1] = 1., 1.
    >>> data[2:-2,2], data[2:-2,-2] = 2., 2.
    >>> data[2,2:-2], data[-3,2:-2] = 2., 2.
    >>> data[3,3:-2] = 3.
    >>> data = np.roll(data, 4, 1)

    We need to set up the interpolator object. Here, we must also specify the
    coordinates of the knots to use.

    >>> lats, lons = np.meshgrid(theta, phi)
    >>> knotst, knotsp = theta.copy(), phi.copy()
    >>> knotst[0] += .0001
    >>> knotst[-1] -= .0001
    >>> knotsp[0] += .0001
    >>> knotsp[-1] -= .0001
    >>> lut = LSQSphereBivariateSpline(lats.ravel(), lons.ravel(),
    ...                                data.T.ravel(), knotst, knotsp)

    As a first test, we'll see what the algorithm returns when run on the
    input coordinates

    >>> data_orig = lut(theta, phi)

    Finally we interpolate the data to a finer grid

    >>> fine_lats = np.linspace(0., np.pi, 70)
    >>> fine_lons = np.linspace(0., 2*np.pi, 90)
    >>> data_lsq = lut(fine_lats, fine_lons)

    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(131)
    >>> ax1.imshow(data, interpolation='nearest')
    >>> ax2 = fig.add_subplot(132)
    >>> ax2.imshow(data_orig, interpolation='nearest')
    >>> ax3 = fig.add_subplot(133)
    >>> ax3.imshow(data_lsq, interpolation='nearest')
    >>> plt.show()

    """

    def __init__(self, theta, phi, r, tt, tp, w=None, eps=1e-16):
        theta, phi, r = (np.asarray(theta), np.asarray(phi), np.asarray(r))
        tt, tp = (np.asarray(tt), np.asarray(tp))
        if not ((0.0 <= theta).all() and (theta <= np.pi).all()):
            raise ValueError('theta should be between [0, pi]')
        if not ((0.0 <= phi).all() and (phi <= 2 * np.pi).all()):
            raise ValueError('phi should be between [0, 2pi]')
        if not ((0.0 < tt).all() and (tt < np.pi).all()):
            raise ValueError('tt should be between (0, pi)')
        if not ((0.0 < tp).all() and (tp < 2 * np.pi).all()):
            raise ValueError('tp should be between (0, 2pi)')
        if w is not None:
            w = np.asarray(w)
            if not (w >= 0.0).all():
                raise ValueError('w should be positive')
        if not 0.0 < eps < 1.0:
            raise ValueError('eps should be between (0, 1)')
        nt_, np_ = (8 + len(tt), 8 + len(tp))
        tt_, tp_ = (zeros((nt_,), float), zeros((np_,), float))
        tt_[4:-4], tp_[4:-4] = (tt, tp)
        tt_[-4:], tp_[-4:] = (np.pi, 2.0 * np.pi)
        tt_, tp_, c, fp, ier = dfitpack.spherfit_lsq(theta, phi, r, tt_, tp_, w=w, eps=eps)
        if ier > 0:
            message = _spherefit_messages.get(ier, 'ier=%s' % ier)
            raise ValueError(message)
        self.fp = fp
        self.tck = (tt_, tp_, c)
        self.degrees = (3, 3)

    def __call__(self, theta, phi, dtheta=0, dphi=0, grid=True):
        theta = np.asarray(theta)
        phi = np.asarray(phi)
        if phi.size > 0 and (phi.min() < 0.0 or phi.max() > 2.0 * np.pi):
            raise ValueError('requested phi out of bounds.')
        return SphereBivariateSpline.__call__(self, theta, phi, dtheta=dtheta, dphi=dphi, grid=grid)