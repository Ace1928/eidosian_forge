import cupy
from cupyx.scipy._lib._util import _asarray_validated, float_factorial
class _Interpolator1DWithDerivatives(_Interpolator1D):

    def derivatives(self, x, der=None):
        """Evaluate many derivatives of the polynomial at the point x.

        The function produce an array of all derivative values at
        the point x.

        Parameters
        ----------
        x : cupy.ndarray
            Point or points at which to evaluate the derivatives
        der : int or None, optional
            How many derivatives to extract; None for all potentially
            nonzero derivatives (that is a number equal to the number
            of points). This number includes the function value as 0th
            derivative

        Returns
        -------
        d : cupy.ndarray
            Array with derivatives; d[j] contains the jth derivative.
            Shape of d[j] is determined by replacing the interpolation
            axis in the original array with the shape of x

        """
        x, x_shape = self._prepare_x(x)
        y = self._evaluate_derivatives(x, der)
        y = y.reshape((y.shape[0],) + x_shape + self._y_extra_shape)
        if self._y_axis != 0 and x_shape != ():
            nx = len(x_shape)
            ny = len(self._y_extra_shape)
            s = [0] + list(range(nx + 1, nx + self._y_axis + 1)) + list(range(1, nx + 1)) + list(range(nx + 1 + self._y_axis, nx + ny + 1))
            y = y.transpose(s)
        return y

    def derivative(self, x, der=1):
        """Evaluate one derivative of the polynomial at the point x

        Parameters
        ----------
        x : cupy.ndarray
            Point or points at which to evaluate the derivatives
        der : integer, optional
            Which derivative to extract. This number includes the
            function value as 0th derivative

        Returns
        -------
        d : cupy.ndarray
            Derivative interpolated at the x-points. Shape of d is
            determined by replacing the interpolation axis in the
            original array with the shape of x

        Notes
        -----
        This is computed by evaluating all derivatives up to the desired
        one (using self.derivatives()) and then discarding the rest.

        """
        x, x_shape = self._prepare_x(x)
        y = self._evaluate_derivatives(x, der + 1)
        return self._finish_y(y[der], x_shape)