import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Var, Set, TransformationFactory
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
def test_reclassification(self):
    m = ConcreteModel()
    m.t = ContinuousSet(bounds=(0, 1))
    m.x = ContinuousSet(bounds=(5, 10))
    m.s = Set(initialize=[1, 2, 3])
    m.v = Var(m.t)
    m.v2 = Var(m.s, m.t)
    m.v3 = Var(m.x, m.t)
    m.dv = DerivativeVar(m.v)
    m.dv2 = DerivativeVar(m.v2, wrt=(m.t, m.t))
    m.dv3 = DerivativeVar(m.v3, wrt=m.x)
    TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t)
    self.assertTrue(m.dv.ctype is Var)
    self.assertTrue(m.dv2.ctype is Var)
    self.assertTrue(m.dv.is_fully_discretized())
    self.assertTrue(m.dv2.is_fully_discretized())
    self.assertTrue(m.dv3.ctype is DerivativeVar)
    self.assertFalse(m.dv3.is_fully_discretized())
    TransformationFactory('dae.collocation').apply_to(m, wrt=m.x)
    self.assertTrue(m.dv3.ctype is Var)
    self.assertTrue(m.dv3.is_fully_discretized())