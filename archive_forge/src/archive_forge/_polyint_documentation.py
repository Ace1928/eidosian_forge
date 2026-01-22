import cupy
from cupyx.scipy._lib._util import _asarray_validated, float_factorial
Evaluate the interpolating polynomial at the points x.

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

        