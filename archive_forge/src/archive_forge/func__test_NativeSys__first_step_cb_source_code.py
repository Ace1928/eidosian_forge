from __future__ import print_function, absolute_import, division
from collections import defaultdict
import numpy as np
from pyodesys.util import import_
from pyodesys.core import integrate_chained
from pyodesys.symbolic import ScaledSys, TransformedSys, symmetricsys, PartiallySolvedSystem, get_logexp
from pyodesys.tests.test_core import (
from pyodesys.tests.bateman import bateman_full  # analytic, never mind the details
from pyodesys.tests.test_symbolic import decay_rhs, decay_dydt_factory, _get_decay3
def _test_NativeSys__first_step_cb_source_code(NativeSys, log10myconst, should_succeed, forgive=20, **kwargs):
    dec3 = _get_decay3()
    odesys = NativeSys.from_other(dec3, namespace_override={'p_first_step': 'return good_const()*y[0];', 'p_anon': 'double good_const(){ return std::pow(10, %.5g); }' % log10myconst}, namespace_extend={'p_includes': ['<cmath>']})
    y0, k = ([0.7, 0, 0], [1e+23, 2, 3.0])
    xout, yout, info = odesys.integrate(5, y0, k, integrator='native', **kwargs)
    ref = np.array(bateman_full(y0, k, xout - xout[0], exp=np.exp)).T
    allclose_kw = dict(atol=kwargs['atol'] * forgive, rtol=kwargs['rtol'] * forgive)
    if should_succeed is None:
        assert not np.allclose(yout, ref, **allclose_kw)
    else:
        assert info['success'] == should_succeed
        info['nfev'] > 10 and info['nfev'] > 1 and (info['time_cpu'] < 100)
        if should_succeed:
            assert np.allclose(yout, ref, **allclose_kw)