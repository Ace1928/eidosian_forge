import math
from warnings import warn
import numpy as np
from numpy.linalg import inv
from scipy import optimize, spatial
class EllipseModel(BaseModel):
    """Total least squares estimator for 2D ellipses.

    The functional model of the ellipse is::

        xt = xc + a*cos(theta)*cos(t) - b*sin(theta)*sin(t)
        yt = yc + a*sin(theta)*cos(t) + b*cos(theta)*sin(t)
        d = sqrt((x - xt)**2 + (y - yt)**2)

    where ``(xt, yt)`` is the closest point on the ellipse to ``(x, y)``. Thus
    d is the shortest distance from the point to the ellipse.

    The estimator is based on a least squares minimization. The optimal
    solution is computed directly, no iterations are required. This leads
    to a simple, stable and robust fitting method.

    The ``params`` attribute contains the parameters in the following order::

        xc, yc, a, b, theta

    Attributes
    ----------
    params : tuple
        Ellipse model parameters in the following order `xc`, `yc`, `a`, `b`,
        `theta`.

    Examples
    --------

    >>> xy = EllipseModel().predict_xy(np.linspace(0, 2 * np.pi, 25),
    ...                                params=(10, 15, 8, 4, np.deg2rad(30)))
    >>> ellipse = EllipseModel()
    >>> ellipse.estimate(xy)
    True
    >>> np.round(ellipse.params, 2)
    array([10.  , 15.  ,  8.  ,  4.  ,  0.52])
    >>> np.round(abs(ellipse.residuals(xy)), 5)
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0.])
    """

    def estimate(self, data):
        """Estimate ellipse model from data using total least squares.

        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.


        References
        ----------
        .. [1] Halir, R.; Flusser, J. "Numerically stable direct least squares
               fitting of ellipses". In Proc. 6th International Conference in
               Central Europe on Computer Graphics and Visualization.
               WSCG (Vol. 98, pp. 125-132).

        """
        _check_data_dim(data, dim=2)
        float_type = np.promote_types(data.dtype, np.float32)
        data = data.astype(float_type, copy=False)
        origin = data.mean(axis=0)
        data = data - origin
        scale = data.std()
        if scale < np.finfo(float_type).tiny:
            warn('Standard deviation of data is too small to estimate ellipse with meaningful precision.', category=RuntimeWarning, stacklevel=2)
            return False
        data /= scale
        x = data[:, 0]
        y = data[:, 1]
        D1 = np.vstack([x ** 2, x * y, y ** 2]).T
        D2 = np.vstack([x, y, np.ones_like(x)]).T
        S1 = D1.T @ D1
        S2 = D1.T @ D2
        S3 = D2.T @ D2
        C1 = np.array([[0.0, 0.0, 2.0], [0.0, -1.0, 0.0], [2.0, 0.0, 0.0]])
        try:
            M = inv(C1) @ (S1 - S2 @ inv(S3) @ S2.T)
        except np.linalg.LinAlgError:
            return False
        eig_vals, eig_vecs = np.linalg.eig(M)
        cond = 4 * np.multiply(eig_vecs[0, :], eig_vecs[2, :]) - np.power(eig_vecs[1, :], 2)
        a1 = eig_vecs[:, cond > 0]
        if 0 in a1.shape or len(a1.ravel()) != 3:
            return False
        a, b, c = a1.ravel()
        a2 = -inv(S3) @ S2.T @ a1
        d, f, g = a2.ravel()
        b /= 2.0
        d /= 2.0
        f /= 2.0
        x0 = (c * d - b * f) / (b ** 2.0 - a * c)
        y0 = (a * f - b * d) / (b ** 2.0 - a * c)
        numerator = a * f ** 2 + c * d ** 2 + g * b ** 2 - 2 * b * d * f - a * c * g
        term = np.sqrt((a - c) ** 2 + 4 * b ** 2)
        denominator1 = (b ** 2 - a * c) * (term - (a + c))
        denominator2 = (b ** 2 - a * c) * (-term - (a + c))
        width = np.sqrt(2 * numerator / denominator1)
        height = np.sqrt(2 * numerator / denominator2)
        phi = 0.5 * np.arctan(2.0 * b / (a - c))
        if a > c:
            phi += 0.5 * np.pi
        if width < height:
            width, height = (height, width)
            phi += np.pi / 2
        phi %= np.pi
        params = np.nan_to_num([x0, y0, width, height, phi]).real
        params[:4] *= scale
        params[:2] += origin
        self.params = tuple((float(p) for p in params))
        return True

    def residuals(self, data):
        """Determine residuals of data to model.

        For each point the shortest distance to the ellipse is returned.

        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.

        Returns
        -------
        residuals : (N,) array
            Residual for each data point.

        """
        _check_data_dim(data, dim=2)
        xc, yc, a, b, theta = self.params
        ctheta = math.cos(theta)
        stheta = math.sin(theta)
        x = data[:, 0]
        y = data[:, 1]
        N = data.shape[0]

        def fun(t, xi, yi):
            ct = math.cos(np.squeeze(t))
            st = math.sin(np.squeeze(t))
            xt = xc + a * ctheta * ct - b * stheta * st
            yt = yc + a * stheta * ct + b * ctheta * st
            return (xi - xt) ** 2 + (yi - yt) ** 2
        residuals = np.empty((N,), dtype=np.float64)
        t0 = np.arctan2(y - yc, x - xc) - theta
        for i in range(N):
            xi = x[i]
            yi = y[i]
            t, _ = optimize.leastsq(fun, t0[i], args=(xi, yi))
            residuals[i] = np.sqrt(fun(t, xi, yi))
        return residuals

    def predict_xy(self, t, params=None):
        """Predict x- and y-coordinates using the estimated model.

        Parameters
        ----------
        t : array
            Angles in circle in radians. Angles start to count from positive
            x-axis to positive y-axis in a right-handed system.
        params : (5,) array, optional
            Optional custom parameter set.

        Returns
        -------
        xy : (..., 2) array
            Predicted x- and y-coordinates.

        """
        if params is None:
            params = self.params
        xc, yc, a, b, theta = params
        ct = np.cos(t)
        st = np.sin(t)
        ctheta = math.cos(theta)
        stheta = math.sin(theta)
        x = xc + a * ctheta * ct - b * stheta * st
        y = yc + a * stheta * ct + b * ctheta * st
        return np.concatenate((x[..., None], y[..., None]), axis=t.ndim)