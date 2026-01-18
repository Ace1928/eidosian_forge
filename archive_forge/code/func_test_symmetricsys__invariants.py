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
@requires('sym')
def test_symmetricsys__invariants():
    yn, pn = ('x y z'.split(), 'a b'.split())
    odesys = SymbolicSys.from_callback(lambda t, y, p: {'x': -p['a'] * y['x'], 'y': -p['b'] * y['y'] + p['a'] * y['x'], 'z': p['b'] * y['y']}, names=yn, param_names=pn, dep_by_name=True, par_by_name=True, linear_invariants=[[1, 1, 1]], linear_invariant_names=['mass-conservation'], indep_name='t')
    assert odesys.linear_invariants.tolist() == [[1, 1, 1]]
    assert odesys.linear_invariant_names == ['mass-conservation']
    assert odesys.nonlinear_invariants is None
    assert odesys.nonlinear_invariant_names is None
    logexp = get_logexp()
    LogLogSys = symmetricsys(logexp, logexp)
    tsys = LogLogSys.from_other(odesys)
    assert tsys.linear_invariants is None
    assert tsys.linear_invariant_names is None
    assert len(tsys.nonlinear_invariants) == 1
    E = odesys.be.exp
    assert tsys.nonlinear_invariants[0] - sum((E(odesys[k]) for k in yn)) == 0
    assert tsys.nonlinear_invariant_names == ['mass-conservation']