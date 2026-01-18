import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import check_available_solvers
from pyomo.environ import (
from pyomo.core.plugins.transform.standard_form import StandardForm
from pyomo.core.plugins.transform.nonnegative_transform import NonNegativeTransformation
def test_relax_integrality_only_active_blocks(self):
    self.model.x = Var(domain=NonNegativeIntegers)
    self.model.b = Block()
    self.model.b.x = Var(domain=Binary)
    self.model.b.y = Var(domain=Integers, bounds=(-3, 2))
    instance = self.model.create_instance()
    instance.b.deactivate()
    relax_integrality = TransformationFactory('core.relax_integer_vars')
    relax_integrality.apply_to(instance, transform_deactivated_blocks=False)
    self.assertIs(instance.b.x.domain, Binary)
    self.assertIs(instance.b.y.domain, Integers)
    self.assertIs(instance.x.domain, Reals)
    self.assertEqual(instance.x.lb, 0)
    self.assertIsNone(instance.x.ub)