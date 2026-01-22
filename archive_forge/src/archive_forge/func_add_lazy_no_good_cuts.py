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
def add_lazy_no_good_cuts(self, var_values, mindtpy_solver, config, opt, feasible=False):
    """Adds no-good cuts.

        Add the no-good cuts through Cplex inherent function self.add().

        Parameters
        ----------
        var_values : list
            The variable values of the incumbent solution, used to generate the cut.
        mindtpy_solver : object
            The mindtpy solver class.
        config : ConfigBlock
            The specific configurations for MindtPy.
        opt : SolverFactory
            The cplex_persistent solver.
        feasible : bool, optional
            Whether the integer combination yields a feasible or infeasible NLP, by default False.

        Raises
        ------
        ValueError
            The value of binary variable is not 0 or 1.
        """
    if not config.add_no_good_cuts:
        return
    config.logger.debug('Adding no-good cuts')
    with time_code(mindtpy_solver.timing, 'No-good cut generation'):
        m = mindtpy_solver.mip
        MindtPy = m.MindtPy_utils
        int_tol = config.integer_tolerance
        binary_vars = [v for v in MindtPy.variable_list if v.is_binary()]
        for var, val in zip(MindtPy.variable_list, var_values):
            if not var.is_binary():
                continue
            var.stale = True
            var.set_value(val, skip_validation=True)
        for v in binary_vars:
            if value(abs(v - 1)) > int_tol and value(abs(v)) > int_tol:
                raise ValueError('Binary {} = {} is not 0 or 1'.format(v.name, value(v)))
        if not binary_vars:
            return
        pyomo_no_good_cut = sum((1 - v for v in binary_vars if value(abs(v - 1)) <= int_tol)) + sum((v for v in binary_vars if value(abs(v)) <= int_tol))
        cplex_no_good_rhs = generate_standard_repn(pyomo_no_good_cut).constant
        cplex_no_good_cut, _ = opt._get_expr_from_pyomo_expr(pyomo_no_good_cut)
        self.add(constraint=cplex.SparsePair(ind=cplex_no_good_cut.variables, val=cplex_no_good_cut.coefficients), sense='G', rhs=1 - cplex_no_good_rhs)