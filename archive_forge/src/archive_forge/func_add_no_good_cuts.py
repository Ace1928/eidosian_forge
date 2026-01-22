from math import copysign
from pyomo.core import minimize, value
import pyomo.core.expr as EXPR
from pyomo.contrib.gdpopt.util import time_code
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, MCPP_Error
def add_no_good_cuts(target_model, var_values, config, timing, mip_iter=0, cb_opt=None):
    """Adds no-good cuts.

    This adds an no-good cuts to the no_good_cuts ConstraintList, which is not activated by default.
    However, it may be activated as needed in certain situations or for certain values of option flags.


    Parameters
    ----------
    target_model : Block
        The model to add no-good cuts to.
    var_values : list
        Variable values of the current solution, used to generate the cut.
    config : ConfigBlock
        The specific configurations for MindtPy.
    timing : Timing
        Timing.
    mip_iter : Int, optional
        MIP iteration counter.
    cb_opt : SolverFactory, optional
        Gurobi_persistent solver, by default None.

    Raises
    ------
    ValueError
        The value of binary variable is not 0 or 1.
    """
    if not config.add_no_good_cuts:
        return
    with time_code(timing, 'no_good cut generation'):
        config.logger.debug('Adding no-good cuts')
        m = target_model
        MindtPy = m.MindtPy_utils
        int_tol = config.integer_tolerance
        binary_vars = [v for v in MindtPy.variable_list if v.is_binary()]
        for var, val in zip(MindtPy.variable_list, var_values):
            if not var.is_binary():
                continue
            var.set_value(val, skip_validation=True)
        for v in binary_vars:
            if value(abs(v - 1)) > int_tol and value(abs(v)) > int_tol:
                raise ValueError('Binary {} = {} is not 0 or 1'.format(v.name, value(v)))
        if not binary_vars:
            return
        int_cut = sum((1 - v for v in binary_vars if value(abs(v - 1)) <= int_tol)) + sum((v for v in binary_vars if value(abs(v)) <= int_tol)) >= 1
        MindtPy.cuts.no_good_cuts.add(expr=int_cut)
        if config.single_tree and config.mip_solver == 'gurobi_persistent' and (mip_iter > 0) and (cb_opt is not None):
            cb_opt.cbLazy(target_model.MindtPy_utils.cuts.no_good_cuts[len(target_model.MindtPy_utils.cuts.no_good_cuts)])