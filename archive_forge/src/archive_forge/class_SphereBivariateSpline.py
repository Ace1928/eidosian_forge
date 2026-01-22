import warnings
from numpy import zeros, concatenate, ravel, diff, array, ones  # noqa:F401
import numpy as np
from . import _fitpack_impl
from . import dfitpack
class SphereBivariateSpline(_BivariateSplineBase):
    """
    Bivariate spline s(x,y) of degrees 3 on a sphere, calculated from a
    given set of data points (theta,phi,r).

    .. versionadded:: 0.11.0

    See Also
    --------
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    LSQUnivariateSpline :
        a univariate spline using weighted least-squares fitting
    """

    def __call__(self, theta, phi, dtheta=0, dphi=0, grid=True):
        """
        Evaluate the spline or its derivatives at given positions.

        Parameters
        ----------
        theta, phi : array_like
            Input coordinates.

            If `grid` is False, evaluate the spline at points
            ``(theta[i], phi[i]), i=0, ..., len(x)-1``.  Standard
            Numpy broadcasting is obeyed.

            If `grid` is True: evaluate spline at the grid points
            defined by the coordinate arrays theta, phi. The arrays
            must be sorted to increasing order.
            The ordering of axes is consistent with
            ``np.meshgrid(..., indexing="ij")`` and inconsistent with the
            default ordering ``np.meshgrid(..., indexing="xy")``.
        dtheta : int, optional
            Order of theta-derivative

            .. versionadded:: 0.14.0
        dphi : int
            Order of phi-derivative

            .. versionadded:: 0.14.0
        grid : bool
            Whether to evaluate the results on a grid spanned by the
            input arrays, or at points specified by the input arrays.

            .. versionadded:: 0.14.0

        Examples
        --------

        Suppose that we want to use splines to interpolate a bivariate function on a
        sphere. The value of the function is known on a grid of longitudes and
        colatitudes.

        >>> import numpy as np
        >>> from scipy.interpolate import RectSphereBivariateSpline
        >>> def f(theta, phi):
        ...     return np.sin(theta) * np.cos(phi)

        We evaluate the function on the grid. Note that the default indexing="xy"
        of meshgrid would result in an unexpected (transposed) result after
        interpolation.

        >>> thetaarr = np.linspace(0, np.pi, 22)[1:-1]
        >>> phiarr = np.linspace(0, 2 * np.pi, 21)[:-1]
        >>> thetagrid, phigrid = np.meshgrid(thetaarr, phiarr, indexing="ij")
        >>> zdata = f(thetagrid, phigrid)

        We next set up the interpolator and use it to evaluate the function
        on a finer grid.

        >>> rsbs = RectSphereBivariateSpline(thetaarr, phiarr, zdata)
        >>> thetaarr_fine = np.linspace(0, np.pi, 200)
        >>> phiarr_fine = np.linspace(0, 2 * np.pi, 200)
        >>> zdata_fine = rsbs(thetaarr_fine, phiarr_fine)

        Finally we plot the coarsly-sampled input data alongside the
        finely-sampled interpolated data to check that they agree.

        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> ax1 = fig.add_subplot(1, 2, 1)
        >>> ax2 = fig.add_subplot(1, 2, 2)
        >>> ax1.imshow(zdata)
        >>> ax2.imshow(zdata_fine)
        >>> plt.show()
        """
        theta = np.asarray(theta)
        phi = np.asarray(phi)
        if theta.size > 0 and (theta.min() < 0.0 or theta.max() > np.pi):
            raise ValueError('requested theta out of bounds.')
        return _BivariateSplineBase.__call__(self, theta, phi, dx=dtheta, dy=dphi, grid=grid)

    def ev(self, theta, phi, dtheta=0, dphi=0):
        """
        Evaluate the spline at points

        Returns the interpolated value at ``(theta[i], phi[i]),
        i=0,...,len(theta)-1``.

        Parameters
        ----------
        theta, phi : array_like
            Input coordinates. Standard Numpy broadcasting is obeyed.
            The ordering of axes is consistent with
            np.meshgrid(..., indexing="ij") and inconsistent with the
            default ordering np.meshgrid(..., indexing="xy").
        dtheta : int, optional
            Order of theta-derivative

            .. versionadded:: 0.14.0
        dphi : int, optional
            Order of phi-derivative

            .. versionadded:: 0.14.0

        Examples
        --------
        Suppose that we want to use splines to interpolate a bivariate function on a
        sphere. The value of the function is known on a grid of longitudes and
        colatitudes.

        >>> import numpy as np
        >>> from scipy.interpolate import RectSphereBivariateSpline
        >>> def f(theta, phi):
        ...     return np.sin(theta) * np.cos(phi)

        We evaluate the function on the grid. Note that the default indexing="xy"
        of meshgrid would result in an unexpected (transposed) result after
        interpolation.

        >>> thetaarr = np.linspace(0, np.pi, 22)[1:-1]
        >>> phiarr = np.linspace(0, 2 * np.pi, 21)[:-1]
        >>> thetagrid, phigrid = np.meshgrid(thetaarr, phiarr, indexing="ij")
        >>> zdata = f(thetagrid, phigrid)

        We next set up the interpolator and use it to evaluate the function
        at points not on the original grid.

        >>> rsbs = RectSphereBivariateSpline(thetaarr, phiarr, zdata)
        >>> thetainterp = np.linspace(thetaarr[0], thetaarr[-1], 200)
        >>> phiinterp = np.linspace(phiarr[0], phiarr[-1], 200)
        >>> zinterp = rsbs.ev(thetainterp, phiinterp)

        Finally we plot the original data for a diagonal slice through the
        initial grid, and the spline approximation along the same slice.

        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> ax1 = fig.add_subplot(1, 1, 1)
        >>> ax1.plot(np.sin(thetaarr) * np.sin(phiarr), np.diag(zdata), "or")
        >>> ax1.plot(np.sin(thetainterp) * np.sin(phiinterp), zinterp, "-b")
        >>> plt.show()
        """
        return self.__call__(theta, phi, dtheta=dtheta, dphi=dphi, grid=False)