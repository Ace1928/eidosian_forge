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
def _gsl(tout, method, forgive):
    n, p, a = (13, 1, 13)
    atol, rtol = (1e-10, 1e-10)
    y0, k, odesys_dens = get_special_chain(n, p, a)
    dydt = decay_dydt_factory(k)
    odesys_dens = SymbolicSys.from_callback(dydt, len(k) + 1)
    xout, yout, info = odesys_dens.integrate(tout, y0, method=method, integrator='gsl', atol=atol, rtol=rtol)
    check(yout[-1, :], n, p, a, atol, rtol, forgive)