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
class TestSimulator(unittest.TestCase):
    """
    Class for testing the pyomo.DAE simulator
    """

    def setUp(self):
        """
        Setting up testing model
        """
        self.m = m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.v = Var(m.t)
        m.dv = DerivativeVar(m.v)
        m.s = Set(initialize=[1, 2, 3], ordered=True)

    def test_invalid_argument_values(self):
        m = self.m
        m.w = Var(m.t)
        m.y = Var()
        with self.assertRaises(DAE_Error):
            Simulator(m, package='foo')

        def _con(m, i):
            return m.v[i] == m.w[i] ** 2 + m.y
        m.con = Constraint(m.t, rule=_con)
        with self.assertRaises(DAE_Error):
            Simulator(m, package='scipy')
        m.del_component('con')
        m.del_component('con_index')
        m.del_component('w')
        m.del_component('y')

    @unittest.skipIf(not scipy_available, 'Scipy is not available')
    def test_inequality_constraints(self):
        m = self.m

        def _deq(m, i):
            return m.dv[i] >= m.v[i] ** 2 + m.v[i]
        m.deq = Constraint(m.t, rule=_deq)
        mysim = Simulator(m)
        self.assertEqual(len(mysim._diffvars), 0)
        self.assertEqual(len(mysim._derivlist), 0)
        self.assertEqual(len(mysim._rhsdict), 0)

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

    @unittest.skipIf(not scipy_available, 'Scipy is not available')
    def test_separable_diffeq_case4(self):
        m = self.m
        m.w = Var(m.t, m.s)
        m.dw = DerivativeVar(m.w)
        m.p = Param(initialize=5)
        m.mp = Param(initialize=5, mutable=True)
        m.y = Var()
        t = IndexTemplate(m.t)

        def _deqv(m, i):
            return m.v[i] ** 2 + m.v[i] == m.p * m.dv[i]
        m.deqv = Constraint(m.t, rule=_deqv)

        def _deqw(m, i, j):
            return m.w[i, j] ** 2 + m.w[i, j] == m.p * m.dw[i, j]
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
            return m.v[i] ** 2 + m.v[i] == m.mp * m.dv[i]
        m.deqv = Constraint(m.t, rule=_deqv)

        def _deqw(m, i, j):
            return m.w[i, j] ** 2 + m.w[i, j] == m.y * m.dw[i, j]
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

    @unittest.skipIf(not scipy_available, 'Scipy is not available')
    def test_separable_diffeq_case5(self):
        m = self.m
        m.w = Var(m.t, m.s)
        m.dw = DerivativeVar(m.w)
        m.p = Param(initialize=5)
        m.mp = Param(initialize=5, mutable=True)
        m.y = Var()
        t = IndexTemplate(m.t)

        def _deqv(m, i):
            return m.dv[i] + m.y == m.v[i] ** 2 + m.v[i]
        m.deqv = Constraint(m.t, rule=_deqv)

        def _deqw(m, i, j):
            return m.y + m.dw[i, j] == m.w[i, j] ** 2 + m.w[i, j]
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
            return m.mp + m.dv[i] == m.v[i] ** 2 + m.v[i]
        m.deqv = Constraint(m.t, rule=_deqv)

        def _deqw(m, i, j):
            return m.dw[i, j] + m.p == m.w[i, j] ** 2 + m.w[i, j]
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

    @unittest.skipIf(not scipy_available, 'Scipy is not available')
    def test_separable_diffeq_case6(self):
        m = self.m
        m.w = Var(m.t, m.s)
        m.dw = DerivativeVar(m.w)
        m.p = Param(initialize=5)
        m.mp = Param(initialize=5, mutable=True)
        m.y = Var()
        t = IndexTemplate(m.t)

        def _deqv(m, i):
            return m.v[i] ** 2 + m.v[i] == m.dv[i] + m.y
        m.deqv = Constraint(m.t, rule=_deqv)

        def _deqw(m, i, j):
            return m.w[i, j] ** 2 + m.w[i, j] == m.y + m.dw[i, j]
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
            return m.v[i] ** 2 + m.v[i] == m.mp + m.dv[i]
        m.deqv = Constraint(m.t, rule=_deqv)

        def _deqw(m, i, j):
            return m.w[i, j] ** 2 + m.w[i, j] == m.dw[i, j] + m.p
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

    @unittest.skipIf(not scipy_available, 'Scipy is not available')
    def test_separable_diffeq_case8(self):
        m = self.m
        m.w = Var(m.t, m.s)
        m.dw = DerivativeVar(m.w)
        m.p = Param(initialize=5)
        m.mp = Param(initialize=5, mutable=True)
        m.y = Var()
        t = IndexTemplate(m.t)

        def _deqv(m, i):
            return -m.dv[i] == m.v[i] ** 2 + m.v[i]
        m.deqv = Constraint(m.t, rule=_deqv)

        def _deqw(m, i, j):
            return -m.dw[i, j] == m.w[i, j] ** 2 + m.w[i, j]
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

    @unittest.skipIf(not scipy_available, 'Scipy is not available')
    def test_separable_diffeq_case9(self):
        m = self.m
        m.w = Var(m.t, m.s)
        m.dw = DerivativeVar(m.w)
        m.p = Param(initialize=5)
        m.mp = Param(initialize=5, mutable=True)
        m.y = Var()
        t = IndexTemplate(m.t)

        def _deqv(m, i):
            return m.v[i] ** 2 + m.v[i] == -m.dv[i]
        m.deqv = Constraint(m.t, rule=_deqv)

        def _deqw(m, i, j):
            return m.w[i, j] ** 2 + m.w[i, j] == -m.dw[i, j]
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

    @unittest.skipIf(not scipy_available, 'Scipy is not available')
    def test_sim_initialization_multi_index(self):
        m = self.m
        m.w1 = Var(m.t, m.s)
        m.dw1 = DerivativeVar(m.w1)
        m.w2 = Var(m.s, m.t)
        m.dw2 = DerivativeVar(m.w2)
        m.w3 = Var([0, 1], m.t, m.s)
        m.dw3 = DerivativeVar(m.w3)
        t = IndexTemplate(m.t)

        def _deq1(m, t, s):
            return m.dw1[t, s] == m.w1[t, s]
        m.deq1 = Constraint(m.t, m.s, rule=_deq1)

        def _deq2(m, s, t):
            return m.dw2[s, t] == m.w2[s, t]
        m.deq2 = Constraint(m.s, m.t, rule=_deq2)

        def _deq3(m, i, t, s):
            return m.dw3[i, t, s] == m.w1[t, s] + m.w2[i + 1, t]
        m.deq3 = Constraint([0, 1], m.t, m.s, rule=_deq3)
        mysim = Simulator(m)
        self.assertIs(mysim._contset, m.t)
        self.assertEqual(len(mysim._diffvars), 12)
        self.assertTrue(_GetItemIndexer(m.w1[t, 1]) in mysim._diffvars)
        self.assertTrue(_GetItemIndexer(m.w1[t, 3]) in mysim._diffvars)
        self.assertTrue(_GetItemIndexer(m.w2[1, t]) in mysim._diffvars)
        self.assertTrue(_GetItemIndexer(m.w2[3, t]) in mysim._diffvars)
        self.assertTrue(_GetItemIndexer(m.w3[0, t, 1]) in mysim._diffvars)
        self.assertTrue(_GetItemIndexer(m.w3[1, t, 3]) in mysim._diffvars)
        self.assertEqual(len(mysim._derivlist), 12)
        self.assertTrue(_GetItemIndexer(m.dw1[t, 1]) in mysim._derivlist)
        self.assertTrue(_GetItemIndexer(m.dw1[t, 3]) in mysim._derivlist)
        self.assertTrue(_GetItemIndexer(m.dw2[1, t]) in mysim._derivlist)
        self.assertTrue(_GetItemIndexer(m.dw2[3, t]) in mysim._derivlist)
        self.assertTrue(_GetItemIndexer(m.dw3[0, t, 1]) in mysim._derivlist)
        self.assertTrue(_GetItemIndexer(m.dw3[1, t, 3]) in mysim._derivlist)
        self.assertEqual(len(mysim._templatemap), 6)
        self.assertTrue(_GetItemIndexer(m.w1[t, 1]) in mysim._templatemap)
        self.assertTrue(_GetItemIndexer(m.w1[t, 3]) in mysim._templatemap)
        self.assertTrue(_GetItemIndexer(m.w2[1, t]) in mysim._templatemap)
        self.assertTrue(_GetItemIndexer(m.w2[3, t]) in mysim._templatemap)
        self.assertFalse(_GetItemIndexer(m.w3[0, t, 1]) in mysim._templatemap)
        self.assertFalse(_GetItemIndexer(m.w3[1, t, 3]) in mysim._templatemap)
        self.assertEqual(len(mysim._rhsdict), 12)
        self.assertTrue(isinstance(mysim._rhsdict[_GetItemIndexer(m.dw1[t, 1])], Param))
        self.assertTrue(isinstance(mysim._rhsdict[_GetItemIndexer(m.dw1[t, 3])], Param))
        self.assertTrue(isinstance(mysim._rhsdict[_GetItemIndexer(m.dw2[1, t])], Param))
        self.assertTrue(isinstance(mysim._rhsdict[_GetItemIndexer(m.dw2[3, t])], Param))
        self.assertTrue(isinstance(mysim._rhsdict[_GetItemIndexer(m.dw3[0, t, 1])], EXPR.SumExpression))
        self.assertTrue(isinstance(mysim._rhsdict[_GetItemIndexer(m.dw3[1, t, 3])], EXPR.SumExpression))
        self.assertEqual(mysim._rhsdict[_GetItemIndexer(m.dw1[t, 1])].name, "'w1[{t},1]'")
        self.assertEqual(mysim._rhsdict[_GetItemIndexer(m.dw1[t, 3])].name, "'w1[{t},3]'")
        self.assertEqual(mysim._rhsdict[_GetItemIndexer(m.dw2[1, t])].name, "'w2[1,{t}]'")
        self.assertEqual(mysim._rhsdict[_GetItemIndexer(m.dw2[3, t])].name, "'w2[3,{t}]'")
        self.assertEqual(len(mysim._rhsfun(0, [0] * 12)), 12)
        self.assertIsNone(mysim._tsim)
        self.assertIsNone(mysim._simsolution)
        m.del_component('deq1')
        m.del_component('deq1_index')
        m.del_component('deq2')
        m.del_component('deq2_index')
        m.del_component('deq3')
        m.del_component('deq3_index')

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

    def test_non_supported_single_index(self):
        m = ConcreteModel()
        with self.assertRaises(DAE_Error):
            Simulator(m)
        m = ConcreteModel()
        m.s = ContinuousSet(bounds=(0, 10))
        m.t = ContinuousSet(bounds=(0, 5))
        with self.assertRaises(DAE_Error):
            Simulator(m)
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        with self.assertRaises(DAE_Error):
            Simulator(m)
        m = self.m

        def _diffeq(m, t):
            return m.dv[t] == m.v[t] ** 2 + m.v[t]
        m.con1 = Constraint(m.t, rule=_diffeq)
        m.con2 = Constraint(m.t, rule=_diffeq)
        with self.assertRaises(DAE_Error):
            Simulator(m)
        m.del_component('con1')
        m.del_component('con2')
        m = self.m

        def _diffeq(m, t):
            return m.dv[t] == m.dv[t] + m.v[t] ** 2
        m.con1 = Constraint(m.t, rule=_diffeq)
        with self.assertRaises(DAE_Error):
            Simulator(m)
        m.del_component('con1')

    @unittest.skipIf(not scipy_available, 'Scipy is not available')
    def test_non_supported_multi_index(self):
        m = self.m
        m.v2 = Var(m.t, m.s)
        m.v3 = Var(m.s, m.t)
        m.dv2 = DerivativeVar(m.v2)
        m.dv3 = DerivativeVar(m.v3)

        def _diffeq(m, t, s):
            return m.dv2[t, s] == m.v2[t, s] ** 2 + m.v2[t, s]
        m.con1 = Constraint(m.t, m.s, rule=_diffeq)
        m.con2 = Constraint(m.t, m.s, rule=_diffeq)
        with self.assertRaises(DAE_Error):
            Simulator(m)
        m.del_component('con1')
        m.del_component('con2')
        m.del_component('con1_index')
        m.del_component('con2_index')

        def _diffeq(m, s, t):
            return m.dv3[s, t] == m.v3[s, t] ** 2 + m.v3[s, t]
        m.con1 = Constraint(m.s, m.t, rule=_diffeq)
        m.con2 = Constraint(m.s, m.t, rule=_diffeq)
        with self.assertRaises(DAE_Error):
            Simulator(m)
        m.del_component('con1')
        m.del_component('con2')
        m.del_component('con1_index')
        m.del_component('con2_index')

        def _diffeq(m, t, s):
            return m.dv2[t, s] == m.dv2[t, s] + m.v2[t, s] ** 2
        m.con1 = Constraint(m.t, m.s, rule=_diffeq)
        with self.assertRaises(DAE_Error):
            Simulator(m)
        m.del_component('con1')
        m.del_component('con1_index')

        def _diffeq(m, s, t):
            return m.dv3[s, t] == m.dv3[s, t] + m.v3[s, t] ** 2
        m.con1 = Constraint(m.s, m.t, rule=_diffeq)
        with self.assertRaises(DAE_Error):
            Simulator(m)
        m.del_component('con1')
        m.del_component('con1_index')

    def test_scipy_unsupported(self):
        m = self.m
        m.a = Var(m.t)

        def _diffeq(m, t):
            return 0 == m.v[t] ** 2 + m.a[t]
        m.con = Constraint(m.t, rule=_diffeq)
        with self.assertRaises(DAE_Error):
            Simulator(m, package='scipy')
        m.del_component('con')

    @unittest.skipIf(not scipy_available, 'Scipy is not available')
    def test_time_indexed_algebraic(self):
        m = self.m
        m.a = Var(m.t)

        def _diffeq(m, t):
            return m.dv[t] == m.v[t] ** 2 + m.a[t]
        m.con = Constraint(m.t, rule=_diffeq)
        mysim = Simulator(m)
        t = IndexTemplate(m.t)
        self.assertEqual(len(mysim._algvars), 1)
        self.assertTrue(_GetItemIndexer(m.a[t]) in mysim._algvars)
        self.assertEqual(len(mysim._alglist), 0)
        m.del_component('con')

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

    @unittest.skipIf(not casadi_available, 'casadi not available')
    def test_nonRHS_vars(self):
        m = self.m
        m.v2 = Var(m.t)
        m.dv2 = DerivativeVar(m.v2)
        m.p = Param(initialize=5)
        t = IndexTemplate(m.t)

        def _con(m, t):
            return m.dv2[t] == 10 + m.p
        m.con = Constraint(m.t, rule=_con)
        mysim = Simulator(m, package='casadi')
        self.assertEqual(len(mysim._templatemap), 1)
        self.assertEqual(mysim._diffvars[0], _GetItemIndexer(m.v2[t]))
        m.del_component('con')