import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import check_available_solvers
from pyomo.environ import (
from pyomo.core.plugins.transform.standard_form import StandardForm
from pyomo.core.plugins.transform.nonnegative_transform import NonNegativeTransformation
def test_nonnegativity_transformation_2(self):
    self.model.S = RangeSet(0, 10)
    self.model.T = Set(initialize=['foo', 'bar'])
    self.model.x1 = Var(bounds=(-3, 3))
    self.model.y1 = Var(self.model.S, bounds=(-3, 3))
    self.model.z1 = Var(self.model.S, self.model.T, bounds=(-3, 3))

    def boundsRule(*args):
        return (-4, 4)
    self.model.x2 = Var(bounds=boundsRule)
    self.model.y2 = Var(self.model.S, bounds=boundsRule)
    self.model.z2 = Var(self.model.S, self.model.T, bounds=boundsRule)
    self.model.x3 = Var(domain=NegativeReals)
    self.model.y3 = Var(self.model.S, domain=NegativeIntegers)
    self.model.z3 = Var(self.model.S, self.model.T, domain=Reals)

    def domainRule(*args):
        if len(args) == 1 or args[0] == 0:
            return NonNegativeReals
        elif args[0] == 1:
            return NonNegativeIntegers
        elif args[0] == 2:
            return NonPositiveReals
        elif args[0] == 3:
            return NonPositiveIntegers
        elif args[0] == 4:
            return NegativeReals
        elif args[0] == 5:
            return NegativeIntegers
        elif args[0] == 6:
            return PositiveReals
        elif args[0] == 7:
            return PositiveIntegers
        elif args[0] == 8:
            return Reals
        elif args[0] == 9:
            return Integers
        elif args[0] == 10:
            return Binary
        else:
            return NonNegativeReals
    self.model.x4 = Var(domain=domainRule)
    self.model.y4 = Var(self.model.S, domain=domainRule)
    self.model.z4 = Var(self.model.S, self.model.T, domain=domainRule)
    instance = self.model.create_instance()
    xfrm = TransformationFactory('core.nonnegative_vars')
    transformed = xfrm.create_using(instance)
    for c in ('x', 'y', 'z'):
        for n in ('1', '2', '3', '4'):
            var = transformed.__getattribute__(c + n)
            for ndx in var.index_set():
                self.assertTrue(self.nonnegativeBounds(var[ndx]))