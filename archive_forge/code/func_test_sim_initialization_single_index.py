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
def test_sim_initialization_single_index(self):
    m = self.m
    m.w = Var(m.t)
    m.dw = DerivativeVar(m.w)
    t = IndexTemplate(m.t)

    def _deq1(m, i):
        return m.dv[i] == m.v[i]
    m.deq1 = Constraint(m.t, rule=_deq1)

    def _deq2(m, i):
        return m.dw[i] == m.v[i]
    m.deq2 = Constraint(m.t, rule=_deq2)
    mysim = Simulator(m)
    self.assertIs(mysim._contset, m.t)
    self.assertEqual(len(mysim._diffvars), 2)
    self.assertEqual(mysim._diffvars[0], _GetItemIndexer(m.v[t]))
    self.assertEqual(mysim._diffvars[1], _GetItemIndexer(m.w[t]))
    self.assertEqual(len(mysim._derivlist), 2)
    self.assertEqual(mysim._derivlist[0], _GetItemIndexer(m.dv[t]))
    self.assertEqual(mysim._derivlist[1], _GetItemIndexer(m.dw[t]))
    self.assertEqual(len(mysim._templatemap), 1)
    self.assertTrue(_GetItemIndexer(m.v[t]) in mysim._templatemap)
    self.assertFalse(_GetItemIndexer(m.w[t]) in mysim._templatemap)
    self.assertEqual(len(mysim._rhsdict), 2)
    self.assertTrue(isinstance(mysim._rhsdict[_GetItemIndexer(m.dv[t])], Param))
    self.assertEqual(mysim._rhsdict[_GetItemIndexer(m.dv[t])].name, "'v[{t}]'")
    self.assertTrue(isinstance(mysim._rhsdict[_GetItemIndexer(m.dw[t])], Param))
    self.assertEqual(mysim._rhsdict[_GetItemIndexer(m.dw[t])].name, "'v[{t}]'")
    self.assertEqual(len(mysim._rhsfun(0, [0, 0])), 2)
    self.assertIsNone(mysim._tsim)
    self.assertIsNone(mysim._simsolution)
    m.del_component('deq1')
    m.del_component('deq2')
    m.del_component('dw')
    m.del_component('w')