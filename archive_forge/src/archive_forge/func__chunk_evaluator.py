import math
import warnings
from itertools import combinations_with_replacement
import cupy as cp
def _chunk_evaluator(self, x, y, shift, scale, coeffs, memory_budget=1000000):
    """
        Evaluate the interpolation.

        Parameters
        ----------
        x : (Q, N) float ndarray
            array of points on which to evaluate
        y: (P, N) float ndarray
            array of points on which we know function values
        shift: (N, ) ndarray
            Domain shift used to create the polynomial matrix.
        scale : (N,) float ndarray
            Domain scaling used to create the polynomial matrix.
        coeffs: (P+R, S) float ndarray
            Coefficients in front of basis functions

        Returns
        -------
        (Q, S) float ndarray
        Interpolated array
        """
    nx, ndim = x.shape
    nnei = len(y)
    chunksize = memory_budget // (self.powers.shape[0] + nnei) + 1
    if chunksize <= nx:
        out = cp.empty((nx, self.d.shape[1]), dtype=float)
        for i in range(0, nx, chunksize):
            vec = _build_evaluation_coefficients(x[i:i + chunksize, :], y, self.kernel, self.epsilon, self.powers, shift, scale)
            out[i:i + chunksize, :] = cp.dot(vec, coeffs)
    else:
        vec = _build_evaluation_coefficients(x, y, self.kernel, self.epsilon, self.powers, shift, scale)
        out = cp.dot(vec, coeffs)
    return out