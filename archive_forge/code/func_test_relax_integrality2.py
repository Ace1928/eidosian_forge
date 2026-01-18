import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import check_available_solvers
from pyomo.environ import (
from pyomo.core.plugins.transform.standard_form import StandardForm
from pyomo.core.plugins.transform.nonnegative_transform import NonNegativeTransformation
def test_relax_integrality2(self):
    self.model.A = RangeSet(1, 4)
    self.model.a = Var([1, 2, 3], dense=True)
    self.model.b = Var([1, 2, 3], within=self.model.A, dense=True)
    self.model.c = Var([1, 2, 3], within=NonNegativeIntegers, dense=True)
    self.model.d = Var([1, 2, 3], within=Integers, bounds=(-2, 3), dense=True)
    self.model.e = Var([1, 2, 3], within=Boolean, dense=True)
    self.model.f = Var([1, 2, 3], domain=Boolean, dense=True)
    instance = self.model.create_instance()
    xfrm = TransformationFactory('core.relax_integer_vars')
    rinst = xfrm.create_using(instance)
    self.assertEqual(type(rinst.a[1].domain), RealSet)
    self.assertEqual(type(rinst.b[1].domain), RealSet)
    self.assertEqual(type(rinst.c[1].domain), RealSet)
    self.assertEqual(type(rinst.d[1].domain), RealSet)
    self.assertEqual(type(rinst.e[1].domain), RealSet)
    self.assertEqual(type(rinst.f[1].domain), RealSet)
    self.assertEqual(rinst.a[1].bounds, instance.a[1].bounds)
    self.assertEqual(rinst.b[1].bounds, instance.b[1].bounds)
    self.assertEqual(rinst.c[1].bounds, instance.c[1].bounds)
    self.assertEqual(rinst.d[1].bounds, instance.d[1].bounds)
    self.assertEqual(rinst.e[1].bounds, instance.e[1].bounds)
    self.assertEqual(rinst.f[1].bounds, instance.f[1].bounds)