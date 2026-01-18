from __future__ import (absolute_import, division, print_function)
import math
from collections import OrderedDict
import pytest
import numpy as np
from .. import ODESys, OdeSys, chained_parameter_variation  # OdeSys deprecated
from ..core import integrate_chained
from ..util import requires, pycvodes_klu
@requires('scipy')
def test_pre_post_processors():
    """
    y(x) = A * exp(-k * x)
    dy(x)/dx = -k * A * exp(-k * x) = -k * y(x)

    First transformation:
    v = y/y(x0)
    u = k*x
    ===>
        v(u) = exp(-u)
        dv(u)/du = -v(u)

    Second transformation:
    s = ln(v)
    r = u
    ===>
        s(r) = -r
        ds(r)/dr = -1
    """

    def pre1(x, y, p):
        return (x * p[0], y / y[0], [p[0], y[0]])

    def post1(x, y, p):
        return (x / p[0], y * p[1], [p[0]])

    def pre2(x, y, p):
        return (x, np.log(y), p)

    def post2(x, y, p):
        return (x, np.exp(y), p)

    def dsdr(x, y, p):
        return [-1]
    odesys = ODESys(dsdr, pre_processors=(pre1, pre2), post_processors=(post2, post1))
    k = 3.7
    A = 42
    tend = 7
    xout, yout, info = odesys.integrate(np.asarray([0, tend]), np.asarray([A]), [k], atol=1e-12, rtol=1e-12, name='vode', method='adams')
    yref = A * np.exp(-k * xout)
    assert np.allclose(yout.flatten(), yref)
    assert np.allclose(info['internal_yout'].flatten(), -info['internal_xout'])