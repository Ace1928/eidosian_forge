import os
import random
from ..lp_diff import load_and_compare_lp_baseline
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, Block, ComponentMap
def test_row_ordering(self):
    model = ConcreteModel()
    model.a = Var()
    components = {}
    components['obj'] = Objective(expr=model.a)
    components['con1'] = Constraint(expr=model.a >= 0)
    components['con2'] = Constraint(expr=model.a <= 1)
    components['con3'] = Constraint(expr=(0, model.a, 1))
    components['con4'] = Constraint([1, 2], rule=lambda m, i: model.a == i)
    random_order = list(components.keys())
    random.shuffle(random_order)
    for key in random_order:
        model.add_component(key, components[key])
    row_order = ComponentMap()
    row_order[model.con1] = 100
    row_order[model.con2] = 2
    row_order[model.con3] = 1
    row_order[model.con4[1]] = 0
    row_order[model.con4[2]] = -1
    self._check_baseline(model, row_order=row_order)