import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, ConcreteModel, Set, Var, Piecewise, Constraint
def test_unsorted_pw_pts(self):
    model = ConcreteModel()
    model.range = Var()
    model.x = Var(bounds=(-1, 1))
    args = (model.range, model.x)
    keywords = {'pw_pts': [-1, 0, 1], 'pw_constr_type': 'EQ', 'f_rule': lambda model, x: x ** 2}
    model.con = Piecewise(*args, **keywords)
    try:
        keywords['pw_pts'] = [0, -1, 1]
        model.con3 = Piecewise(*args, **keywords)
    except Exception:
        pass
    else:
        self.fail('Piecewise should fail when initialized with unsorted domain points.')