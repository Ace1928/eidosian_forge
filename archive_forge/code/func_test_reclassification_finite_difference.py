import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Var, Set, TransformationFactory, Expression
from pyomo.dae import ContinuousSet, Integral
from pyomo.dae.diffvar import DAE_Error
from pyomo.repn import generate_standard_repn
def test_reclassification_finite_difference(self):
    m = ConcreteModel()
    m.t = ContinuousSet(bounds=(0, 1))
    m.x = ContinuousSet(bounds=(5, 10))
    m.s = Set(initialize=[1, 2, 3])
    m.v = Var(m.t)
    m.v2 = Var(m.s, m.t)
    m.v3 = Var(m.t, m.x)

    def _int1(m, t):
        return m.v[t]
    m.int1 = Integral(m.t, rule=_int1)

    def _int2(m, s, t):
        return m.v2[s, t]
    m.int2 = Integral(m.s, m.t, wrt=m.t, rule=_int2)

    def _int3(m, t, x):
        return m.v3[t, x]
    m.int3 = Integral(m.t, m.x, wrt=m.t, rule=_int3)

    def _int4(m, x):
        return m.int3[x]
    m.int4 = Integral(m.x, wrt=m.x, rule=_int4)
    self.assertFalse(m.int1.is_fully_discretized())
    self.assertFalse(m.int2.is_fully_discretized())
    self.assertFalse(m.int3.is_fully_discretized())
    self.assertFalse(m.int4.is_fully_discretized())
    TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t)
    self.assertTrue(m.int1.is_fully_discretized())
    self.assertTrue(m.int2.is_fully_discretized())
    self.assertFalse(m.int3.is_fully_discretized())
    self.assertFalse(m.int4.is_fully_discretized())
    self.assertTrue(m.int1.ctype is Integral)
    self.assertTrue(m.int2.ctype is Integral)
    self.assertTrue(m.int3.ctype is Integral)
    self.assertTrue(m.int4.ctype is Integral)
    TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.x)
    self.assertTrue(m.int3.is_fully_discretized())
    self.assertTrue(m.int4.is_fully_discretized())
    self.assertTrue(m.int1.ctype is Expression)
    self.assertTrue(m.int2.ctype is Expression)
    self.assertTrue(m.int3.ctype is Expression)
    self.assertTrue(m.int4.ctype is Expression)