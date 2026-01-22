import itertools
import logging
from pyomo.common.collections import ComponentMap
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.gc_manager import PauseGC
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.base import Reference, TransformationFactory
import pyomo.core.expr as EXPR
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.plugins.bigm_mixin import (
from pyomo.gdp.plugins.gdp_to_mip_transformation import GDP_to_MIP_Transformation
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import get_gdp_tree, _to_dict
from pyomo.network import Port
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.repn import generate_standard_repn
from weakref import ref as weakref_ref
def _warn_for_active_suffix(self, suffix, disjunct, active_disjuncts, Ms):
    if suffix.local_name == 'BigM':
        logger.debug("Found active 'BigM' Suffix on '{0}'. The multiple bigM transformation does not currently support specifying M's with Suffixes and is ignoring this Suffix.".format(disjunct.name))
    elif suffix.local_name == 'LocalVars':
        pass
    else:
        raise GDP_Error("Found active Suffix '{0}' on Disjunct '{1}'. The multiple bigM transformation does not support this Suffix.".format(suffix.name, disjunct.name))