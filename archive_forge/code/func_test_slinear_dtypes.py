from numpy.testing import (assert_, assert_equal, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
from numpy import mgrid, pi, sin, ogrid, poly1d, linspace
import numpy as np
from scipy.interpolate import (interp1d, interp2d, lagrange, PPoly, BPoly,
from scipy.special import poch, gamma
from scipy.interpolate import _ppoly
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
from scipy.integrate import nquad
from scipy.special import binom
def test_slinear_dtypes(self):
    dt_r = [np.float16, np.float32, np.float64]
    dt_rc = dt_r + [np.complex64, np.complex128]
    spline_kinds = ['slinear', 'zero', 'quadratic', 'cubic']
    for dtx in dt_r:
        x = np.arange(0, 10, dtype=dtx)
        for dty in dt_rc:
            y = np.exp(-x / 3.0).astype(dty)
            for dtn in dt_r:
                xnew = x.astype(dtn)
                for kind in spline_kinds:
                    f = interp1d(x, y, kind=kind, bounds_error=False)
                    assert_allclose(f(xnew), y, atol=1e-07, err_msg=f'{dtx}, {dty} {dtn}')