from pyomo.core import (
from pyomo.core.base import TransformationFactory, _VarData
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.common.config import ConfigBlock, ConfigValue, NonNegativeFloat
from pyomo.common.modeling import unique_component_name
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.opt import TerminationCondition
import logging
def _nonneg_scalar_multiply_linear_constraint_error_msg(self, cons, coef):
    return 'The do_integer_arithmetic flag was set to True, but the lower bound of %s is non-integer within the specified tolerance, with value %s. \nPlease set do_integer_arithmetic=False, increase integer_tolerance, or make your data integer.' % (cons['body'].to_expression() >= cons['lower'], coef)