import logging
from pyomo.common.config import (
from pyomo.contrib.gdpopt.util import _DoNothing, a_logger
from pyomo.common.deprecation import deprecation_warning
def _add_fp_configs(CONFIG):
    """Adds the feasibility pump-related configurations.

    Parameters
    ----------
    CONFIG : ConfigBlock
        The specific configurations for MindtPy.
    """
    CONFIG.declare('fp_cutoffdecr', ConfigValue(default=0.1, domain=PositiveFloat, description='Additional relative decrement of cutoff value for the original objective function.'))
    CONFIG.declare('fp_iteration_limit', ConfigValue(default=20, domain=PositiveInt, description='Feasibility pump iteration limit', doc='Number of maximum iterations in the feasibility pump methods.'))
    CONFIG.declare('fp_projcuts', ConfigValue(default=True, description='Whether to add cut derived from regularization of MIP solution onto NLP feasible set.', domain=bool))
    CONFIG.declare('fp_transfercuts', ConfigValue(default=True, description='Whether to transfer cuts from the Feasibility Pump MIP to main MIP in selected strategy (all except from the round in which the FP MIP became infeasible).', domain=bool))
    CONFIG.declare('fp_projzerotol', ConfigValue(default=0.0001, domain=PositiveFloat, description='Tolerance on when to consider optimal value of regularization problem as zero, which may trigger the solution of a Sub-NLP.'))
    CONFIG.declare('fp_mipgap', ConfigValue(default=0.01, domain=PositiveFloat, description='Optimality tolerance (relative gap) to use for solving MIP regularization problem.'))
    CONFIG.declare('fp_discrete_only', ConfigValue(default=True, description='Only calculate the distance among discrete variables in regularization problems.', domain=bool))
    CONFIG.declare('fp_main_norm', ConfigValue(default='L1', domain=In(['L1', 'L2', 'L_infinity']), description='Different forms of objective function MIP regularization problem.'))
    CONFIG.declare('fp_norm_constraint', ConfigValue(default=True, description='Whether to add the norm constraint to FP-NLP', domain=bool))
    CONFIG.declare('fp_norm_constraint_coef', ConfigValue(default=1, domain=PositiveFloat, description='The coefficient in the norm constraint, correspond to the Beta in the paper.'))