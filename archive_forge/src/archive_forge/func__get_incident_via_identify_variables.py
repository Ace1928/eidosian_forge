from contextlib import nullcontext
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.expr.numvalue import value as pyo_value
from pyomo.repn import generate_standard_repn
from pyomo.util.subsystems import TemporarySubsystemManager
from pyomo.repn.plugins.nl_writer import AMPLRepn
from pyomo.contrib.incidence_analysis.config import (
def _get_incident_via_identify_variables(expr, include_fixed):
    return list(identify_variables(expr, include_fixed=include_fixed))