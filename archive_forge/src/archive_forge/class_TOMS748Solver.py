import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult, _call_callback_maybe_halt
import numpy as np
class TOMS748Solver:
    """Solve f(x, *args) == 0 using Algorithm748 of Alefeld, Potro & Shi.
    """
    _MU = 0.5
    _K_MIN = 1
    _K_MAX = 100

    def __init__(self):
        self.f = None
        self.args = None
        self.function_calls = 0
        self.iterations = 0
        self.k = 2
        self.ab = [np.nan, np.nan]
        self.fab = [np.nan, np.nan]
        self.d = None
        self.fd = None
        self.e = None
        self.fe = None
        self.disp = False
        self.xtol = _xtol
        self.rtol = _rtol
        self.maxiter = _iter

    def configure(self, xtol, rtol, maxiter, disp, k):
        self.disp = disp
        self.xtol = xtol
        self.rtol = rtol
        self.maxiter = maxiter
        self.k = max(k, self._K_MIN)
        if self.k > self._K_MAX:
            msg = 'toms748: Overriding k: ->%d' % self._K_MAX
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
            self.k = self._K_MAX

    def _callf(self, x, error=True):
        """Call the user-supplied function, update book-keeping"""
        fx = self.f(x, *self.args)
        self.function_calls += 1
        if not np.isfinite(fx) and error:
            raise ValueError(f'Invalid function value: f({x:f}) -> {fx} ')
        return fx

    def get_result(self, x, flag=_ECONVERGED):
        """Package the result and statistics into a tuple."""
        return (x, self.function_calls, self.iterations, flag)

    def _update_bracket(self, c, fc):
        return _update_bracket(self.ab, self.fab, c, fc)

    def start(self, f, a, b, args=()):
        """Prepare for the iterations."""
        self.function_calls = 0
        self.iterations = 0
        self.f = f
        self.args = args
        self.ab[:] = [a, b]
        if not np.isfinite(a) or np.imag(a) != 0:
            raise ValueError('Invalid x value: %s ' % a)
        if not np.isfinite(b) or np.imag(b) != 0:
            raise ValueError('Invalid x value: %s ' % b)
        fa = self._callf(a)
        if not np.isfinite(fa) or np.imag(fa) != 0:
            raise ValueError(f'Invalid function value: f({a:f}) -> {fa} ')
        if fa == 0:
            return (_ECONVERGED, a)
        fb = self._callf(b)
        if not np.isfinite(fb) or np.imag(fb) != 0:
            raise ValueError(f'Invalid function value: f({b:f}) -> {fb} ')
        if fb == 0:
            return (_ECONVERGED, b)
        if np.sign(fb) * np.sign(fa) > 0:
            raise ValueError(f'f(a) and f(b) must have different signs, but f({a:e})={fa:e}, f({b:e})={fb:e} ')
        self.fab[:] = [fa, fb]
        return (_EINPROGRESS, sum(self.ab) / 2.0)

    def get_status(self):
        """Determine the current status."""
        a, b = self.ab[:2]
        if np.isclose(a, b, rtol=self.rtol, atol=self.xtol):
            return (_ECONVERGED, sum(self.ab) / 2.0)
        if self.iterations >= self.maxiter:
            return (_ECONVERR, sum(self.ab) / 2.0)
        return (_EINPROGRESS, sum(self.ab) / 2.0)

    def iterate(self):
        """Perform one step in the algorithm.

        Implements Algorithm 4.1(k=1) or 4.2(k=2) in [APS1995]
        """
        self.iterations += 1
        eps = np.finfo(float).eps
        d, fd, e, fe = (self.d, self.fd, self.e, self.fe)
        ab_width = self.ab[1] - self.ab[0]
        c = None
        for nsteps in range(2, self.k + 2):
            if _notclose(self.fab + [fd, fe], rtol=0, atol=32 * eps):
                c0 = _inverse_poly_zero(self.ab[0], self.ab[1], d, e, self.fab[0], self.fab[1], fd, fe)
                if self.ab[0] < c0 < self.ab[1]:
                    c = c0
            if c is None:
                c = _newton_quadratic(self.ab, self.fab, d, fd, nsteps)
            fc = self._callf(c)
            if fc == 0:
                return (_ECONVERGED, c)
            e, fe = (d, fd)
            d, fd = self._update_bracket(c, fc)
        uix = 0 if np.abs(self.fab[0]) < np.abs(self.fab[1]) else 1
        u, fu = (self.ab[uix], self.fab[uix])
        _, A = _compute_divided_differences(self.ab, self.fab, forward=uix == 0, full=False)
        c = u - 2 * fu / A
        if np.abs(c - u) > 0.5 * (self.ab[1] - self.ab[0]):
            c = sum(self.ab) / 2.0
        elif np.isclose(c, u, rtol=eps, atol=0):
            frs = np.frexp(self.fab)[1]
            if frs[uix] < frs[1 - uix] - 50:
                c = (31 * self.ab[uix] + self.ab[1 - uix]) / 32
            else:
                mm = 1 if uix == 0 else -1
                adj = mm * np.abs(c) * self.rtol + mm * self.xtol
                c = u + adj
            if not self.ab[0] < c < self.ab[1]:
                c = sum(self.ab) / 2.0
        fc = self._callf(c)
        if fc == 0:
            return (_ECONVERGED, c)
        e, fe = (d, fd)
        d, fd = self._update_bracket(c, fc)
        if self.ab[1] - self.ab[0] > self._MU * ab_width:
            e, fe = (d, fd)
            z = sum(self.ab) / 2.0
            fz = self._callf(z)
            if fz == 0:
                return (_ECONVERGED, z)
            d, fd = self._update_bracket(z, fz)
        self.d, self.fd = (d, fd)
        self.e, self.fe = (e, fe)
        status, xn = self.get_status()
        return (status, xn)

    def solve(self, f, a, b, args=(), xtol=_xtol, rtol=_rtol, k=2, maxiter=_iter, disp=True):
        """Solve f(x) = 0 given an interval containing a root."""
        self.configure(xtol=xtol, rtol=rtol, maxiter=maxiter, disp=disp, k=k)
        status, xn = self.start(f, a, b, args)
        if status == _ECONVERGED:
            return self.get_result(xn)
        c = _secant(self.ab, self.fab)
        if not self.ab[0] < c < self.ab[1]:
            c = sum(self.ab) / 2.0
        fc = self._callf(c)
        if fc == 0:
            return self.get_result(c)
        self.d, self.fd = self._update_bracket(c, fc)
        self.e, self.fe = (None, None)
        self.iterations += 1
        while True:
            status, xn = self.iterate()
            if status == _ECONVERGED:
                return self.get_result(xn)
            if status == _ECONVERR:
                fmt = 'Failed to converge after %d iterations, bracket is %s'
                if disp:
                    msg = fmt % (self.iterations + 1, self.ab)
                    raise RuntimeError(msg)
                return self.get_result(xn, _ECONVERR)