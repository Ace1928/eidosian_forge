from pyomo.common.errors import InfeasibleConstraintException
import pyomo.common.unittest as unittest
from pyomo.environ import Block, ConcreteModel, Constraint, TransformationFactory
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error
def test_not_enough_info_in_single_disjunction_to_fully_transform_or(self):
    m = ConcreteModel()
    m.d = Disjunct([1, 2, 3, 4])
    m.disj1 = Disjunction(expr=[m.d[1], m.d[2], m.d[3], m.d[4]], xor=False)
    m.d[1].indicator_var = True
    m.d[2].indicator_var = False
    with self.assertRaisesRegex(GDP_Error, "Disjunction 'disj1' does not contain enough Disjuncts with values in their indicator_vars to specify which Disjuncts are True. Cannot fully transform model."):
        reverse = TransformationFactory('gdp.transform_current_disjunctive_state').apply_to(m)