from pyomo.core import ConcreteModel, Var, Objective, Piecewise
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class LP_piecewise_nosuffixes(LP_piecewise):
    description = 'LP_piecewise_nosuffixes'
    test_pickling = False

    def __init__(self):
        LP_piecewise.__init__(self)
        self.disable_suffix_tests = True
        self.add_results('LP_piecewise.json')