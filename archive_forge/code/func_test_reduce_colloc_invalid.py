import pyomo.common.unittest as unittest
from pyomo.environ import Var, Set, ConcreteModel, TransformationFactory, pyomo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from pyomo.repn import generate_standard_repn
from io import StringIO
from pyomo.common.log import LoggingIntercept
from os.path import abspath, dirname, normpath, join
def test_reduce_colloc_invalid(self):
    m = self.m.clone()
    m.u = Var(m.t)
    m2 = m.clone()
    disc = TransformationFactory('dae.collocation')
    disc2 = TransformationFactory('dae.collocation')
    disc.apply_to(m, nfe=5, ncp=3)
    with self.assertRaises(TypeError):
        disc.reduce_collocation_points(m, contset=None)
    with self.assertRaises(TypeError):
        disc.reduce_collocation_points(m, contset=m.s)
    with self.assertRaises(RuntimeError):
        disc2.reduce_collocation_points(m2, contset=m2.t)
    m2.tt = ContinuousSet(bounds=(0, 1))
    disc2.apply_to(m2, wrt=m2.t)
    with self.assertRaises(ValueError):
        disc2.reduce_collocation_points(m2, contset=m2.tt)
    with self.assertRaises(TypeError):
        disc.reduce_collocation_points(m, contset=m.t, var=None)
    with self.assertRaises(TypeError):
        disc.reduce_collocation_points(m, contset=m.t, var=m.s)
    with self.assertRaises(TypeError):
        disc.reduce_collocation_points(m, contset=m.t, var=m.v1, ncp=None)
    with self.assertRaises(ValueError):
        disc.reduce_collocation_points(m, contset=m.t, var=m.v1, ncp=-3)
    with self.assertRaises(ValueError):
        disc.reduce_collocation_points(m, contset=m.t, var=m.v1, ncp=10)
    m.v2 = Var()
    m.v3 = Var(m.s)
    m.v4 = Var(m.s, m.s)
    with self.assertRaises(IndexError):
        disc.reduce_collocation_points(m, contset=m.t, var=m.v2, ncp=1)
    with self.assertRaises(IndexError):
        disc.reduce_collocation_points(m, contset=m.t, var=m.v3, ncp=1)
    with self.assertRaises(IndexError):
        disc.reduce_collocation_points(m, contset=m.t, var=m.v4, ncp=1)
    disc.reduce_collocation_points(m, contset=m.t, var=m.u, ncp=1)
    with self.assertRaises(RuntimeError):
        disc.reduce_collocation_points(m, contset=m.t, var=m.u, ncp=1)