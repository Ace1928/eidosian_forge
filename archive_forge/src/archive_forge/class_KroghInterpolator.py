import cupy
from cupyx.scipy._lib._util import _asarray_validated, float_factorial
class KroghInterpolator(_Interpolator1DWithDerivatives):
    """Interpolating polynomial for a set of points.

    The polynomial passes through all the pairs (xi,yi). One may
    additionally specify a number of derivatives at each point xi;
    this is done by repeating the value xi and specifying the
    derivatives as successive yi values
    Allows evaluation of the polynomial and all its derivatives.
    For reasons of numerical stability, this function does not compute
    the coefficients of the polynomial, although they can be obtained
    by evaluating all the derivatives.

    Parameters
    ----------
    xi : cupy.ndarray, length N
        x-coordinate, must be sorted in increasing order
    yi : cupy.ndarray
        y-coordinate, when a xi occurs two or more times in a row,
        the corresponding yi's represent derivative values
    axis : int, optional
        Axis in the yi array corresponding to the x-coordinate values.

    """

    def __init__(self, xi, yi, axis=0):
        _Interpolator1DWithDerivatives.__init__(self, xi, yi, axis)
        self.xi = xi.astype(cupy.float_)
        self.yi = self._reshape_yi(yi)
        self.n, self.r = self.yi.shape
        c = cupy.zeros((self.n + 1, self.r), dtype=self.dtype)
        c[0] = self.yi[0]
        Vk = cupy.zeros((self.n, self.r), dtype=self.dtype)
        for k in range(1, self.n):
            s = 0
            while s <= k and xi[k - s] == xi[k]:
                s += 1
            s -= 1
            Vk[0] = self.yi[k] / float_factorial(s)
            for i in range(k - s):
                if xi[i] == xi[k]:
                    raise ValueError("Elements if `xi` can't be equal.")
                if s == 0:
                    Vk[i + 1] = (c[i] - Vk[i]) / (xi[i] - xi[k])
                else:
                    Vk[i + 1] = (Vk[i + 1] - Vk[i]) / (xi[i] - xi[k])
            c[k] = Vk[k - s]
        self.c = c

    def _evaluate(self, x):
        pi = 1
        p = cupy.zeros((len(x), self.r), dtype=self.dtype)
        p += self.c[0, cupy.newaxis, :]
        for k in range(1, self.n):
            w = x - self.xi[k - 1]
            pi = w * pi
            p += pi[:, cupy.newaxis] * self.c[k]
        return p

    def _evaluate_derivatives(self, x, der=None):
        n = self.n
        r = self.r
        if der is None:
            der = self.n
        pi = cupy.zeros((n, len(x)))
        w = cupy.zeros((n, len(x)))
        pi[0] = 1
        p = cupy.zeros((len(x), self.r), dtype=self.dtype)
        p += self.c[0, cupy.newaxis, :]
        for k in range(1, n):
            w[k - 1] = x - self.xi[k - 1]
            pi[k] = w[k - 1] * pi[k - 1]
            p += pi[k, :, cupy.newaxis] * self.c[k]
        cn = cupy.zeros((max(der, n + 1), len(x), r), dtype=self.dtype)
        cn[:n + 1, :, :] += self.c[:n + 1, cupy.newaxis, :]
        cn[0] = p
        for k in range(1, n):
            for i in range(1, n - k + 1):
                pi[i] = w[k + i - 1] * pi[i - 1] + pi[i]
                cn[k] = cn[k] + pi[i, :, cupy.newaxis] * cn[k + i]
            cn[k] *= float_factorial(k)
        cn[n, :, :] = 0
        return cn[:der]