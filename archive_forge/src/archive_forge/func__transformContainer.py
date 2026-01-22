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
def _transformContainer(self, obj):
    """Find all disjuncts in the container and transform them."""
    for disjunct in obj.component_data_objects(ctype=Disjunct, active=True, descend_into=True):
        _bool = disjunct.indicator_var
        if _bool.value is None:
            raise GDP_Error("The value of the indicator_var of Disjunct '%s' is None. All indicator_vars must have values before calling 'fix_disjuncts'." % disjunct.name)
        elif _bool.value:
            disjunct.indicator_var.fix(True)
            self._transformContainer(disjunct)
        else:
            disjunct.deactivate()
    for disjunction in obj.component_data_objects(ctype=Disjunction, active=True, descend_into=True):
        self._transformDisjunctionData(disjunction)