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
def test_sim_initialization_multi_index2(self):
    m = self.m
    m.s2 = Set(initialize=[(1, 1), (2, 2)])
    m.w1 = Var(m.t, m.s2)
    m.dw1 = DerivativeVar(m.w1)
    m.w2 = Var(m.s2, m.t)
    m.dw2 = DerivativeVar(m.w2)
    m.w3 = Var([0, 1], m.t, m.s2)
    m.dw3 = DerivativeVar(m.w3)
    t = IndexTemplate(m.t)

    def _deq1(m, t, i, j):
        return m.dw1[t, i, j] == m.w1[t, i, j]
    m.deq1 = Constraint(m.t, m.s2, rule=_deq1)

    def _deq2(m, *idx):
        return m.dw2[idx] == m.w2[idx]
    m.deq2 = Constraint(m.s2, m.t, rule=_deq2)

    def _deq3(m, i, t, j, k):
        return m.dw3[i, t, j, k] == m.w1[t, j, k] + m.w2[j, k, t]
    m.deq3 = Constraint([0, 1], m.t, m.s2, rule=_deq3)
    mysim = Simulator(m)
    self.assertIs(mysim._contset, m.t)
    self.assertEqual(len(mysim._diffvars), 8)
    self.assertTrue(_GetItemIndexer(m.w1[t, 1, 1]) in mysim._diffvars)
    self.assertTrue(_GetItemIndexer(m.w1[t, 2, 2]) in mysim._diffvars)
    self.assertTrue(_GetItemIndexer(m.w2[1, 1, t]) in mysim._diffvars)
    self.assertTrue(_GetItemIndexer(m.w2[2, 2, t]) in mysim._diffvars)
    self.assertTrue(_GetItemIndexer(m.w3[0, t, 1, 1]) in mysim._diffvars)
    self.assertTrue(_GetItemIndexer(m.w3[1, t, 2, 2]) in mysim._diffvars)
    self.assertEqual(len(mysim._derivlist), 8)
    self.assertTrue(_GetItemIndexer(m.dw1[t, 1, 1]) in mysim._derivlist)
    self.assertTrue(_GetItemIndexer(m.dw1[t, 2, 2]) in mysim._derivlist)
    self.assertTrue(_GetItemIndexer(m.dw2[1, 1, t]) in mysim._derivlist)
    self.assertTrue(_GetItemIndexer(m.dw2[2, 2, t]) in mysim._derivlist)
    self.assertTrue(_GetItemIndexer(m.dw3[0, t, 1, 1]) in mysim._derivlist)
    self.assertTrue(_GetItemIndexer(m.dw3[1, t, 2, 2]) in mysim._derivlist)
    self.assertEqual(len(mysim._templatemap), 4)
    self.assertTrue(_GetItemIndexer(m.w1[t, 1, 1]) in mysim._templatemap)
    self.assertTrue(_GetItemIndexer(m.w1[t, 2, 2]) in mysim._templatemap)
    self.assertTrue(_GetItemIndexer(m.w2[1, 1, t]) in mysim._templatemap)
    self.assertTrue(_GetItemIndexer(m.w2[2, 2, t]) in mysim._templatemap)
    self.assertFalse(_GetItemIndexer(m.w3[0, t, 1, 1]) in mysim._templatemap)
    self.assertFalse(_GetItemIndexer(m.w3[1, t, 2, 2]) in mysim._templatemap)
    self.assertEqual(len(mysim._rhsdict), 8)
    self.assertTrue(isinstance(mysim._rhsdict[_GetItemIndexer(m.dw1[t, 1, 1])], Param))
    self.assertTrue(isinstance(mysim._rhsdict[_GetItemIndexer(m.dw1[t, 2, 2])], Param))
    self.assertTrue(isinstance(mysim._rhsdict[_GetItemIndexer(m.dw2[1, 1, t])], Param))
    self.assertTrue(isinstance(mysim._rhsdict[_GetItemIndexer(m.dw2[2, 2, t])], Param))
    self.assertTrue(isinstance(mysim._rhsdict[_GetItemIndexer(m.dw3[0, t, 1, 1])], EXPR.SumExpression))
    self.assertTrue(isinstance(mysim._rhsdict[_GetItemIndexer(m.dw3[1, t, 2, 2])], EXPR.SumExpression))
    self.assertEqual(mysim._rhsdict[_GetItemIndexer(m.dw1[t, 1, 1])].name, "'w1[{t},1,1]'")
    self.assertEqual(mysim._rhsdict[_GetItemIndexer(m.dw1[t, 2, 2])].name, "'w1[{t},2,2]'")
    self.assertEqual(mysim._rhsdict[_GetItemIndexer(m.dw2[1, 1, t])].name, "'w2[1,1,{t}]'")
    self.assertEqual(mysim._rhsdict[_GetItemIndexer(m.dw2[2, 2, t])].name, "'w2[2,2,{t}]'")
    self.assertEqual(len(mysim._rhsfun(0, [0] * 8)), 8)
    self.assertIsNone(mysim._tsim)
    self.assertIsNone(mysim._simsolution)
    m.del_component('deq1')
    m.del_component('deq1_index')
    m.del_component('deq2')
    m.del_component('deq2_index')
    m.del_component('deq3')
    m.del_component('deq3_index')