from contextlib import nullcontext
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.expr.numvalue import value as pyo_value
from pyomo.repn import generate_standard_repn
from pyomo.util.subsystems import TemporarySubsystemManager
from pyomo.repn.plugins.nl_writer import AMPLRepn
from pyomo.contrib.incidence_analysis.config import (
def _get_incident_via_standard_repn(expr, include_fixed, linear_only, compute_values=False):
    if include_fixed:
        to_unfix = [var for var in identify_variables(expr, include_fixed=True) if var.fixed]
        context = TemporarySubsystemManager(to_unfix=to_unfix)
    else:
        context = nullcontext()
    with context:
        repn = generate_standard_repn(expr, compute_values=compute_values, quadratic=False)
    linear_vars = []
    for var, coef in zip(repn.linear_vars, repn.linear_coefs):
        try:
            value = pyo_value(coef)
        except ValueError as err:
            if 'No value for uninitialized NumericValue' not in str(err):
                raise err
            value = None
        if value != 0:
            linear_vars.append(var)
    if linear_only:
        nl_var_id_set = set((id(var) for var in repn.nonlinear_vars))
        return [var for var in linear_vars if id(var) not in nl_var_id_set]
    else:
        variables = linear_vars + list(repn.nonlinear_vars)
        unique_variables = []
        id_set = set()
        for var in variables:
            v_id = id(var)
            if v_id not in id_set:
                id_set.add(v_id)
                unique_variables.append(var)
        return unique_variables