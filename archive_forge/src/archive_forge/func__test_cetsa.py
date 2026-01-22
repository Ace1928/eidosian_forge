from __future__ import print_function, absolute_import, division
from collections import defaultdict, OrderedDict
from itertools import product
import math
import numpy as np
import pytest
from .. import ODESys
from ..core import integrate_auto_switch, chained_parameter_variation
from ..symbolic import SymbolicSys, ScaledSys, symmetricsys, PartiallySolvedSystem, get_logexp, _group_invariants
from ..util import requires, pycvodes_double, pycvodes_klu
from .bateman import bateman_full  # analytic, never mind the details
from .test_core import vdp_f
from . import _cetsa
def _test_cetsa(y0, params, extra=False, stepx=1, **kwargs):
    from ._cetsa import _get_cetsa_odesys
    molar_unitless = 1000000000.0
    t0, tend = (1e-16, 180)
    odesys = _get_cetsa_odesys(molar_unitless, False)
    tsys = _get_cetsa_odesys(molar_unitless, True)
    if y0.ndim == 1:
        tout = [t0, tend]
    elif y0.ndim == 2:
        tout = np.asarray([(t0, tend)] * y0.shape[0])
    comb_res = integrate_auto_switch([tsys, odesys], {'nsteps': [500 * stepx, 20 * stepx]}, tout, y0 / molar_unitless, params, return_on_error=True, autorestart=2, **kwargs)
    if isinstance(comb_res, list):
        for r in comb_res:
            assert r.info['success']
            assert r.info['nfev'] > 10
    else:
        assert comb_res.info['success']
        assert comb_res.info['nfev'] > 10
    if extra:
        with pytest.raises(RuntimeError):
            odesys.integrate(np.linspace(t0, tend, 20), y0 / molar_unitless, params, atol=1e-07, rtol=1e-07, nsteps=500, first_step=1e-14, **kwargs)
        res = odesys.integrate(np.linspace(t0, tend, 20), y0 / molar_unitless, params, nsteps=int(38 * 1.1), first_step=1e-14, **kwargs)
        assert np.min(res.yout[-1, :]) < -1e-06
        tres = tsys.integrate([t0, tend], y0 / molar_unitless, params, nsteps=int(1345 * 1.1), **kwargs)
        assert tres.info['success'] is True
        assert tres.info['nfev'] > 100