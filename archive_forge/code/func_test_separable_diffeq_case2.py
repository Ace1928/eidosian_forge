import json
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from pyomo.dae.simulator import (
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.fileutils import import_file
import os
from os.path import abspath, dirname, normpath, join
@unittest.skipIf(not scipy_available, 'Scipy is not available')
def test_separable_diffeq_case2(self):
    m = self.m
    m.w = Var(m.t, m.s)
    m.dw = DerivativeVar(m.w)
    t = IndexTemplate(m.t)

    def _deqv(m, i):
        return m.v[i] ** 2 + m.v[i] == m.dv[i]
    m.deqv = Constraint(m.t, rule=_deqv)

    def _deqw(m, i, j):
        return m.w[i, j] ** 2 + m.w[i, j] == m.dw[i, j]
    m.deqw = Constraint(m.t, m.s, rule=_deqw)
    mysim = Simulator(m)
    self.assertEqual(len(mysim._diffvars), 4)
    self.assertEqual(mysim._diffvars[0], _GetItemIndexer(m.v[t]))
    self.assertEqual(mysim._diffvars[1], _GetItemIndexer(m.w[t, 1]))
    self.assertEqual(mysim._diffvars[2], _GetItemIndexer(m.w[t, 2]))
    self.assertEqual(len(mysim._derivlist), 4)
    self.assertEqual(mysim._derivlist[0], _GetItemIndexer(m.dv[t]))
    self.assertEqual(mysim._derivlist[1], _GetItemIndexer(m.dw[t, 1]))
    self.assertEqual(mysim._derivlist[2], _GetItemIndexer(m.dw[t, 2]))
    self.assertEqual(len(mysim._rhsdict), 4)
    m.del_component('deqv')
    m.del_component('deqw')
    m.del_component('deqv_index')
    m.del_component('deqw_index')
    m.del_component('w')
    m.del_component('dw')