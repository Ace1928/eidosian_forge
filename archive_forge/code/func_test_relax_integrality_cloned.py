import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import check_available_solvers
from pyomo.environ import (
from pyomo.core.plugins.transform.standard_form import StandardForm
from pyomo.core.plugins.transform.nonnegative_transform import NonNegativeTransformation
def test_relax_integrality_cloned(self):
    self.model.A = RangeSet(1, 4)
    self.model.a = Var()
    self.model.b = Var(within=self.model.A)
    self.model.c = Var(within=NonNegativeIntegers)
    self.model.d = Var(within=Integers, bounds=(-2, 3))
    self.model.e = Var(within=Boolean)
    self.model.f = Var(domain=Boolean)
    instance = self.model.create_instance()
    instance_cloned = instance.clone()
    xfrm = TransformationFactory('core.relax_integer_vars')
    rinst = xfrm.create_using(instance_cloned)
    self.assertEqual(type(rinst.a.domain), RealSet)
    self.assertEqual(type(rinst.b.domain), RealSet)
    self.assertEqual(type(rinst.c.domain), RealSet)
    self.assertEqual(type(rinst.d.domain), RealSet)
    self.assertEqual(type(rinst.e.domain), RealSet)
    self.assertEqual(type(rinst.f.domain), RealSet)
    self.assertEqual(rinst.a.bounds, instance_cloned.a.bounds)
    self.assertEqual(rinst.b.bounds, instance_cloned.b.bounds)
    self.assertEqual(rinst.c.bounds, instance_cloned.c.bounds)
    self.assertEqual(rinst.d.bounds, instance_cloned.d.bounds)
    self.assertEqual(rinst.e.bounds, instance_cloned.e.bounds)
    self.assertEqual(rinst.f.bounds, instance_cloned.f.bounds)