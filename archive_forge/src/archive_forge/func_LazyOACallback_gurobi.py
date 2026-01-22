from pyomo.common.dependencies import attempt_import
from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy
from pyomo.contrib.mindtpy.cut_generation import add_oa_cuts, add_no_good_cuts
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, MCPP_Error
from pyomo.repn import generate_standard_repn
import pyomo.core.expr as EXPR
from math import copysign
from pyomo.contrib.mindtpy.util import (
from pyomo.contrib.gdpopt.util import get_main_elapsed_time, time_code
from pyomo.opt import TerminationCondition as tc
from pyomo.core import minimize, value
from pyomo.core.expr import identify_variables
def LazyOACallback_gurobi(cb_m, cb_opt, cb_where, mindtpy_solver, config):
    """This is a Gurobi callback function defined for LP/NLP based B&B algorithm.

    Parameters
    ----------
    cb_m : Pyomo model
        The MIP main problem.
    cb_opt : SolverFactory
        The gurobi_persistent solver.
    cb_where : int
        An enum member of gurobipy.GRB.Callback.
    mindtpy_solver : object
        The mindtpy solver class.
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    if cb_where == gurobipy.GRB.Callback.MIPSOL:
        if mindtpy_solver.should_terminate:
            cb_opt._solver_model.terminate()
            return
        cb_opt.cbGetSolution(vars=cb_m.MindtPy_utils.variable_list)
        handle_lazy_main_feasible_solution_gurobi(cb_m, cb_opt, mindtpy_solver, config)
        if config.add_cuts_at_incumbent:
            if config.strategy == 'OA':
                add_oa_cuts(mindtpy_solver.mip, None, mindtpy_solver.jacobians, mindtpy_solver.objective_sense, mindtpy_solver.mip_constraint_polynomial_degree, mindtpy_solver.mip_iter, config, mindtpy_solver.timing, cb_opt=cb_opt)
        if config.add_regularization is not None and mindtpy_solver.best_solution_found is not None:
            if not mindtpy_solver.dual_bound_improved and (not mindtpy_solver.primal_bound_improved):
                config.logger.debug('The bound and the best found solution have neither been improved.We will skip solving the regularization problem and the Fixed-NLP subproblem')
                mindtpy_solver.primal_bound_improved = False
                return
            if mindtpy_solver.dual_bound != mindtpy_solver.dual_bound_progress[0]:
                mindtpy_solver.add_regularization()
        if mindtpy_solver.bounds_converged() or mindtpy_solver.reached_time_limit():
            cb_opt._solver_model.terminate()
            return
        mindtpy_solver.curr_int_sol = get_integer_solution(mindtpy_solver.fixed_nlp, string_zero=True)
        if mindtpy_solver.curr_int_sol in set(mindtpy_solver.integer_list):
            config.logger.debug('This integer combination has been explored. We will skip solving the Fixed-NLP subproblem.')
            mindtpy_solver.primal_bound_improved = False
            if config.strategy == 'GOA':
                if config.add_no_good_cuts:
                    var_values = list((v.value for v in mindtpy_solver.fixed_nlp.MindtPy_utils.variable_list))
                    add_no_good_cuts(mindtpy_solver.mip, var_values, config, mindtpy_solver.timing, mip_iter=mindtpy_solver.mip_iter, cb_opt=cb_opt)
                return
            elif config.strategy == 'OA':
                begin_index, end_index = mindtpy_solver.integer_solution_to_cuts_index[mindtpy_solver.curr_int_sol]
                for ind in range(begin_index, end_index + 1):
                    cb_opt.cbLazy(mindtpy_solver.mip.MindtPy_utils.cuts.oa_cuts[ind])
                return
        else:
            mindtpy_solver.integer_list.append(mindtpy_solver.curr_int_sol)
            if config.strategy == 'OA':
                cut_ind = len(mindtpy_solver.mip.MindtPy_utils.cuts.oa_cuts)
        fixed_nlp, fixed_nlp_result = mindtpy_solver.solve_subproblem()
        mindtpy_solver.handle_nlp_subproblem_tc(fixed_nlp, fixed_nlp_result, cb_opt)
        if config.strategy == 'OA':
            mindtpy_solver.integer_solution_to_cuts_index[mindtpy_solver.curr_int_sol] = [cut_ind + 1, len(mindtpy_solver.mip.MindtPy_utils.cuts.oa_cuts)]