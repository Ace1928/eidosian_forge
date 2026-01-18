import pyomo.common.unittest as unittest
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.environ import (
from pyomo.repn import generate_standard_repn
from pyomo.gdp import Disjunction
@unittest.skipUnless(sympy_available, 'Sympy not available')
def test_statement_in_Disjunct_with_logical_to_linear(self):
    model = self.create_model()
    model.disj = Disjunction(expr=[[model.x.lor(model.y)], [model.y.lor(model.z)]])
    TransformationFactory('core.logical_to_linear').apply_to(model, targets=model.disj.disjuncts)
    bigmed = TransformationFactory('gdp.bigm').create_using(model)
    self.check_lor_on_disjunct(bigmed, bigmed.disj.disjuncts[0], bigmed.x, bigmed.y)
    self.check_lor_on_disjunct(bigmed, bigmed.disj.disjuncts[1], bigmed.y, bigmed.z)
    TransformationFactory('gdp.hull').apply_to(model)
    self.check_lor_on_disjunct(model, model.disj.disjuncts[0], model.x, model.y)
    self.check_lor_on_disjunct(model, model.disj.disjuncts[1], model.y, model.z)