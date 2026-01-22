import logging
from math import fabs
from pyomo.common.config import ConfigDict, ConfigValue, NonNegativeFloat
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core.base.block import Block
from pyomo.core.expr.numvalue import value
from pyomo.gdp import GDP_Error
from pyomo.gdp.disjunct import Disjunct, Disjunction
from pyomo.gdp.plugins.bigm import BigM_Transformation
def _transformDisjunctionData(self, disjunction):
    logical_sum = sum((value(disj.binary_indicator_var) for disj in disjunction.disjuncts))
    if disjunction.xor and (not logical_sum == 1):
        raise GDP_Error('Disjunction %s violated. Expected 1 disjunct to be active, but %s were active.' % (disjunction.name, logical_sum))
    elif not logical_sum >= 1:
        raise GDP_Error('Disjunction %s violated. Expected at least 1 disjunct to be active, but none were active.')
    else:
        disjunction.deactivate()