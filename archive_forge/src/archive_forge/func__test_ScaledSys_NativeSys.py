from __future__ import print_function, absolute_import, division
from collections import defaultdict
import numpy as np
from pyodesys.util import import_
from pyodesys.core import integrate_chained
from pyodesys.symbolic import ScaledSys, TransformedSys, symmetricsys, PartiallySolvedSystem, get_logexp
from pyodesys.tests.test_core import (
from pyodesys.tests.bateman import bateman_full  # analytic, never mind the details
from pyodesys.tests.test_symbolic import decay_rhs, decay_dydt_factory, _get_decay3
def _test_ScaledSys_NativeSys(NativeSys, nsteps=1000):

    class ScaledNativeSys(ScaledSys, NativeSys):
        pass
    k = k0, k1, k2 = [7.0, 3, 2]
    y0, y1, y2, y3 = sp.symbols('y0 y1 y2 y3', real=True, positive=True)
    l = [(y0, -7 * y0), (y1, 7 * y0 - 3 * y1), (y2, 3 * y1 - 2 * y2), (y3, 2 * y2)]
    ss = ScaledNativeSys(l, dep_scaling=100000000.0)
    y0 = [0] * (len(k) + 1)
    y0[0] = 1
    xout, yout, info = ss.integrate([1e-12, 1], y0, atol=1e-12, rtol=1e-12, nsteps=nsteps)
    ref = np.array(bateman_full(y0, k + [0], xout - xout[0], exp=np.exp)).T
    assert np.allclose(yout, ref, rtol=2e-11, atol=2e-11)