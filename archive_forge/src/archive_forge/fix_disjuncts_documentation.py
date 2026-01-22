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
Find all disjuncts in the container and transform them.