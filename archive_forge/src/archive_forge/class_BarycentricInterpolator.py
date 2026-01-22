import cupy
from cupyx.scipy._lib._util import _asarray_validated, float_factorial
class BarycentricInterpolator(_Interpolator1D):
    """The interpolating polynomial for a set of points.

    Constructs a polynomial that passes through a given set of points.
    Allows evaluation of the polynomial, efficient changing of the y
    values to be interpolated, and updating by adding more x values.
    For reasons of numerical stability, this function does not compute
    the coefficients of the polynomial.
    The value `yi` need to be provided before the function is
    evaluated, but none of the preprocessing depends on them,
    so rapid updates are possible.

    Parameters
    ----------
    xi : cupy.ndarray
        1-D array of x-coordinates of the points the polynomial should
        pass through
    yi : cupy.ndarray, optional
        The y-coordinates of the points the polynomial should pass through.
        If None, the y values will be supplied later via the `set_y` method
    axis : int, optional
        Axis in the yi array corresponding to the x-coordinate values

    See Also
    --------
    scipy.interpolate.BarycentricInterpolator

    """

    def __init__(self, xi, yi=None, axis=0):
        _Interpolator1D.__init__(self, xi, yi, axis)
        self.xi = xi.astype(cupy.float_)
        self.set_yi(yi)
        self.n = len(self.xi)
        self._inv_capacity = 4.0 / (cupy.max(self.xi) - cupy.min(self.xi))
        permute = cupy.random.permutation(self.n)
        inv_permute = cupy.zeros(self.n, dtype=cupy.int32)
        inv_permute[permute] = cupy.arange(self.n)
        self.wi = cupy.zeros(self.n)
        for i in range(self.n):
            dist = self._inv_capacity * (self.xi[i] - self.xi[permute])
            dist[inv_permute[i]] = 1.0
            self.wi[i] = 1.0 / cupy.prod(dist)

    def set_yi(self, yi, axis=None):
        """Update the y values to be interpolated.

        The barycentric interpolation algorithm requires the calculation
        of weights, but these depend only on the xi. The yi can be changed
        at any time.

        Parameters
        ----------
        yi : cupy.ndarray
            The y-coordinates of the points the polynomial should pass
            through. If None, the y values will be supplied later.
        axis : int, optional
            Axis in the yi array corresponding to the x-coordinate values

        """
        if yi is None:
            self.yi = None
            return
        self._set_yi(yi, xi=self.xi, axis=axis)
        self.yi = self._reshape_yi(yi)
        self.n, self.r = self.yi.shape

    def add_xi(self, xi, yi=None):
        """Add more x values to the set to be interpolated.

        The barycentric interpolation algorithm allows easy updating
        by adding more points for the polynomial to pass through.

        Parameters
        ----------
        xi : cupy.ndarray
            The x-coordinates of the points that the polynomial should
            pass through
        yi : cupy.ndarray, optional
            The y-coordinates of the points the polynomial should pass
            through. Should have shape ``(xi.size, R)``; if R > 1 then
            the polynomial is vector-valued
            If `yi` is not given, the y values will be supplied later.
            `yi` should be given if and only if the interpolator has y
            values specified

        """
        if yi is not None:
            if self.yi is None:
                raise ValueError('No previous yi value to update!')
            yi = self._reshape_yi(yi, check=True)
            self.yi = cupy.vstack((self.yi, yi))
        elif self.yi is not None:
            raise ValueError('No update to yi provided!')
        old_n = self.n
        self.xi = cupy.concatenate((self.xi, xi))
        self.n = len(self.xi)
        self.wi **= -1
        old_wi = self.wi
        self.wi = cupy.zeros(self.n)
        self.wi[:old_n] = old_wi
        for j in range(old_n, self.n):
            self.wi[:j] *= self._inv_capacity * (self.xi[j] - self.xi[:j])
            self.wi[j] = cupy.prod(self._inv_capacity * (self.xi[:j] - self.xi[j]))
        self.wi **= -1

    def __call__(self, x):
        """Evaluate the interpolating polynomial at the points x.

        Parameters
        ----------
        x : cupy.ndarray
            Points to evaluate the interpolant at

        Returns
        -------
        y : cupy.ndarray
            Interpolated values. Shape is determined by replacing the
            interpolation axis in the original array with the shape of x

        Notes
        -----
        Currently the code computes an outer product between x and the
        weights, that is, it constructs an intermediate array of size
        N by len(x), where N is the degree of the polynomial.

        """
        return super().__call__(x)

    def _evaluate(self, x):
        if x.size == 0:
            p = cupy.zeros((0, self.r), dtype=self.dtype)
        else:
            c = x[..., cupy.newaxis] - self.xi
            z = c == 0
            c[z] = 1
            c = self.wi / c
            p = cupy.dot(c, self.yi) / cupy.sum(c, axis=-1)[..., cupy.newaxis]
            r = cupy.nonzero(z)
            if len(r) == 1:
                if len(r[0]) > 0:
                    p = self.yi[r[0][0]]
            else:
                p[r[:-1]] = self.yi[r[-1]]
        return p