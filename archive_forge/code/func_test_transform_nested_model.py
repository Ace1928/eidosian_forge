from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.common.log import LoggingIntercept
def test_transform_nested_model(self):
    m = self.create_nested_model()
    bt = TransformationFactory('gdp.bound_pretransformation')
    bt.apply_to(m)
    self.check_nested_model_disjunction(m, bt)
    self.assertEqual(len(list(m.component_data_objects(Constraint, active=True, descend_into=(Block, Disjunct)))), 4)