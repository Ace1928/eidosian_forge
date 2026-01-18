import pyomo.common.unittest as unittest
from pyomo.environ import Var, Set, ConcreteModel, TransformationFactory, pyomo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from pyomo.repn import generate_standard_repn
from io import StringIO
from pyomo.common.log import LoggingIntercept
from os.path import abspath, dirname, normpath, join
def test_reduce_colloc_multi_index(self):
    m = self.m.clone()
    m.u = Var(m.t, m.s)
    m2 = m.clone()
    m3 = m.clone()
    disc = TransformationFactory('dae.collocation')
    disc.apply_to(m, nfe=5, ncp=3)
    disc.reduce_collocation_points(m, contset=m.t, var=m.u, ncp=1)
    self.assertTrue(hasattr(m, 'u_interpolation_constraints'))
    self.assertEqual(len(m.u_interpolation_constraints), 30)
    disc2 = TransformationFactory('dae.collocation')
    disc2.apply_to(m2, wrt=m2.t, nfe=5, ncp=3)
    disc2.reduce_collocation_points(m2, contset=m2.t, var=m2.u, ncp=3)
    self.assertFalse(hasattr(m2, 'u_interpolation_constraints'))
    disc3 = TransformationFactory('dae.collocation')
    disc3.apply_to(m3, wrt=m3.t, nfe=5, ncp=3)
    disc3.reduce_collocation_points(m3, contset=m3.t, var=m3.u, ncp=2)
    self.assertTrue(hasattr(m3, 'u_interpolation_constraints'))
    self.assertEqual(len(m3.u_interpolation_constraints), 15)