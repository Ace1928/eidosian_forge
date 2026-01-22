from __future__ import print_function, absolute_import, division
from collections import defaultdict
import numpy as np
from pyodesys.util import import_
from pyodesys.core import integrate_chained
from pyodesys.symbolic import ScaledSys, TransformedSys, symmetricsys, PartiallySolvedSystem, get_logexp
from pyodesys.tests.test_core import (
from pyodesys.tests.bateman import bateman_full  # analytic, never mind the details
from pyodesys.tests.test_symbolic import decay_rhs, decay_dydt_factory, _get_decay3
def _test_multiple_adaptive_chained(MySys, kw, **kwargs):
    logexp = (sp.log, sp.exp)
    rtol, atol = (1e-14, 1e-12)
    ny = 4
    ks = [[70000000000000.0, 3, 2], [200000.0, 30000.0, 12.7]]
    y0s = [[1.0, 3.0, 2.0, 5.0], [2.0, 1.0, 3.0, 4.0]]
    t0, tend = (1e-16, 7)
    touts = [(t0, tend)] * 2

    class TransformedMySys(TransformedSys, MySys):
        pass
    SS = symmetricsys(logexp, logexp, SuperClass=TransformedMySys)
    tsys = SS.from_callback(decay_rhs, ny, ny - 1)
    osys = MySys.from_callback(decay_rhs, ny, ny - 1)
    comb_res = integrate_chained([tsys, osys], kw, touts, y0s, ks, atol=atol, rtol=rtol, **kwargs)
    for y0, k, res in zip(y0s, ks, comb_res):
        xout, yout = (res.xout, res.yout)
        ref = np.array(bateman_full(y0, k + [0], xout - xout[0], exp=np.exp)).T
        assert np.allclose(yout, ref, rtol=rtol * 1000, atol=atol * 1000)
    for res in comb_res:
        assert 0 <= res.info['time_cpu'] < 100
        assert 0 <= res.info['time_wall'] < 100
        assert res.info['success'] == True