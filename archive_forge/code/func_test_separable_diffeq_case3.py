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
def test_separable_diffeq_case3(self):
    m = self.m
    m.w = Var(m.t, m.s)
    m.dw = DerivativeVar(m.w)
    m.p = Param(initialize=5)
    m.mp = Param(initialize=5, mutable=True)
    m.y = Var()
    t = IndexTemplate(m.t)

    def _deqv(m, i):
        return m.p * m.dv[i] == m.v[i] ** 2 + m.v[i]
    m.deqv = Constraint(m.t, rule=_deqv)

    def _deqw(m, i, j):
        return m.p * m.dw[i, j] == m.w[i, j] ** 2 + m.w[i, j]
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

    def _deqv(m, i):
        return m.mp * m.dv[i] == m.v[i] ** 2 + m.v[i]
    m.deqv = Constraint(m.t, rule=_deqv)

    def _deqw(m, i, j):
        return m.y * m.dw[i, j] == m.w[i, j] ** 2 + m.w[i, j]
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
    m.del_component('p')
    m.del_component('mp')
    m.del_component('y')