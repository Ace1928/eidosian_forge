from pyomo.contrib.gdpopt.util import time_code, get_main_elapsed_time
from pyomo.contrib.mindtpy.util import calc_jacobians
from pyomo.core import ConstraintList
from pyomo.opt import SolverFactory
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_ECP_config
from pyomo.contrib.mindtpy.algorithm_base_class import _MindtPyAlgorithm
from pyomo.contrib.mindtpy.cut_generation import add_ecp_cuts
from pyomo.opt import TerminationCondition as tc
def algorithm_should_terminate(self):
    """Checks if the algorithm should terminate at the given point.

        This function determines whether the algorithm should terminate based on the solver options and progress.
        (Sets the self.results.solver.termination_condition to the appropriate condition, i.e. optimal,
        maxIterations, maxTimeLimit).

        Returns
        -------
        bool
            True if the algorithm should terminate, False otherwise.
        """
    if self.should_terminate:
        if self.primal_bound == self.primal_bound_progress[0]:
            self.results.solver.termination_condition = tc.noSolution
        else:
            self.results.solver.termination_condition = tc.feasible
        return True
    return self.bounds_converged() or self.reached_iteration_limit() or self.reached_time_limit() or self.reached_stalling_limit() or self.all_nonlinear_constraint_satisfied()