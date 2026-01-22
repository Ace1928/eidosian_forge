from math import copysign
from pyomo.core import minimize, value
import pyomo.core.expr as EXPR
from pyomo.contrib.gdpopt.util import time_code
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, MCPP_Error
def add_ecp_cuts(target_model, jacobians, config, timing, linearize_active=True, linearize_violated=True):
    """Linearizes nonlinear constraints. Adds the cuts for the ECP method.

    Parameters
    ----------
    target_model : Pyomo model
        The relaxed linear model.
    jacobians : ComponentMap
        Map nonlinear_constraint --> Map(variable --> jacobian of constraint w.r.t. variable)
    config : ConfigBlock
        The specific configurations for MindtPy.
    timing : Timing
        Timing.
    linearize_active : bool, optional
        Whether to linearize the active nonlinear constraints, by default True.
    linearize_violated : bool, optional
        Whether to linearize the violated nonlinear constraints, by default True.
    """
    with time_code(timing, 'ECP cut generation'):
        for constr in target_model.MindtPy_utils.nonlinear_constraint_list:
            constr_vars = list(EXPR.identify_variables(constr.body))
            jacs = jacobians
            if constr.has_lb() and constr.has_ub():
                config.logger.warning('constraint {} has both a lower and upper bound.\n'.format(constr))
                continue
            if constr.has_ub():
                try:
                    upper_slack = constr.uslack()
                except (ValueError, OverflowError) as e:
                    config.logger.error(e, exc_info=True)
                    config.logger.error('Constraint {} has caused either a ValueError or OverflowError.\n'.format(constr))
                    continue
                if linearize_active and abs(upper_slack) < config.ecp_tolerance or (linearize_violated and upper_slack < 0) or (config.linearize_inactive and upper_slack > 0):
                    if config.add_slack:
                        slack_var = target_model.MindtPy_utils.cuts.slack_vars.add()
                    target_model.MindtPy_utils.cuts.ecp_cuts.add(expr=sum((value(jacs[constr][var]) * (var - var.value) for var in constr_vars)) - (slack_var if config.add_slack else 0) <= upper_slack)
            if constr.has_lb():
                try:
                    lower_slack = constr.lslack()
                except (ValueError, OverflowError) as e:
                    config.logger.error(e, exc_info=True)
                    config.logger.error('Constraint {} has caused either a ValueError or OverflowError.\n'.format(constr))
                    continue
                if linearize_active and abs(lower_slack) < config.ecp_tolerance or (linearize_violated and lower_slack < 0) or (config.linearize_inactive and lower_slack > 0):
                    if config.add_slack:
                        slack_var = target_model.MindtPy_utils.cuts.slack_vars.add()
                    target_model.MindtPy_utils.cuts.ecp_cuts.add(expr=sum((value(jacs[constr][var]) * (var - var.value) for var in constr_vars)) + (slack_var if config.add_slack else 0) >= -lower_slack)