from pyomo.core import (
from pyomo.core.base import TransformationFactory, _VarData
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.common.config import ConfigBlock, ConfigValue, NonNegativeFloat
from pyomo.common.modeling import unique_component_name
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.opt import TerminationCondition
import logging
def vars_to_eliminate_list(x):
    if isinstance(x, (Var, _VarData)):
        if not x.is_indexed():
            return ComponentSet([x])
        ans = ComponentSet()
        for j in x.index_set():
            ans.add(x[j])
        return ans
    elif hasattr(x, '__iter__'):
        ans = ComponentSet()
        for i in x:
            ans.update(vars_to_eliminate_list(i))
        return ans
    else:
        raise ValueError('Expected Var or list of Vars.\n\tReceived %s' % type(x))