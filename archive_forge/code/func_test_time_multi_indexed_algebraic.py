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
def test_time_multi_indexed_algebraic(self):
    m = self.m
    m.v2 = Var(m.t, m.s)
    m.v3 = Var(m.s, m.t)
    m.dv2 = DerivativeVar(m.v2)
    m.dv3 = DerivativeVar(m.v3)
    m.a2 = Var(m.t, m.s)

    def _diffeq(m, t, s):
        return m.dv2[t, s] == m.v2[t, s] ** 2 + m.a2[t, s]
    m.con = Constraint(m.t, m.s, rule=_diffeq)
    m.a3 = Var(m.s, m.t)

    def _diffeq2(m, s, t):
        return m.dv3[s, t] == m.v3[s, t] ** 2 + m.a3[s, t]
    m.con2 = Constraint(m.s, m.t, rule=_diffeq2)
    mysim = Simulator(m)
    t = IndexTemplate(m.t)
    self.assertEqual(len(mysim._algvars), 6)
    self.assertTrue(_GetItemIndexer(m.a2[t, 1]) in mysim._algvars)
    self.assertTrue(_GetItemIndexer(m.a2[t, 3]) in mysim._algvars)
    self.assertTrue(_GetItemIndexer(m.a3[1, t]) in mysim._algvars)
    self.assertTrue(_GetItemIndexer(m.a3[3, t]) in mysim._algvars)
    m.del_component('con')
    m.del_component('con_index')
    m.del_component('con2')
    m.del_component('con2_index')