import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import check_available_solvers
from pyomo.environ import (
from pyomo.core.plugins.transform.standard_form import StandardForm
from pyomo.core.plugins.transform.nonnegative_transform import NonNegativeTransformation
@unittest.skipIf(not 'glpk' in solvers, 'glpk solver is not available')
@unittest.expectedFailure
def test_standard_form_transform_1(self):
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
    self.model.x3 = Var(domain=NegativeReals, bounds=(-10, 10))
    self.model.y3 = Var(self.model.S, domain=NegativeIntegers, bounds=(-10, 10))
    self.model.z3 = Var(self.model.S, self.model.T, domain=Reals, bounds=(-10, 10))

    def domainRule(*args):
        if len(args) == 1:
            arg = 0
        else:
            arg = args[1]
        if len(args) == 1 or arg == 0:
            return NonNegativeReals
        elif arg == 1:
            return NonNegativeIntegers
        elif arg == 2:
            return NonPositiveReals
        elif arg == 3:
            return NonPositiveIntegers
        elif arg == 4:
            return NegativeReals
        elif arg == 5:
            return NegativeIntegers
        elif arg == 6:
            return PositiveReals
        elif arg == 7:
            return PositiveIntegers
        elif arg == 8:
            return Reals
        elif arg == 9:
            return Integers
        elif arg == 10:
            return Binary
        else:
            return Reals
    self.model.x4 = Var(domain=domainRule, bounds=(-10, 10))
    self.model.y4 = Var(self.model.S, domain=domainRule, bounds=(-10, 10))
    self.model.z4 = Var(self.model.S, self.model.T, domain=domainRule, bounds=(-10, 10))

    def objRule(model):
        return sum((5 * sum_product(model.__getattribute__(c + n)) for c in ('x', 'y', 'z') for n in ('1', '2', '3', '4')))
    self.model.obj = Objective(rule=objRule)
    transform = StandardForm()
    instance = self.model.create_instance()
    transformed = transform(instance)
    opt = SolverFactory('glpk')
    instance_sol = opt.solve(instance)
    transformed_sol = opt.solve(transformed)
    self.assertEqual(instance_sol['Solution'][0]['Objective']['obj']['value'], transformed_sol['Solution'][0]['Objective']['obj']['value'])