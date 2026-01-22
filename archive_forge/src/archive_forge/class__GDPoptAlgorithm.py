from io import StringIO
from pyomo.common.collections import Bunch
from pyomo.common.config import ConfigBlock
from pyomo.common.errors import DeveloperError
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.gdpopt.config_options import _add_common_configs
from pyomo.contrib.gdpopt.create_oa_subproblems import (
from pyomo.contrib.gdpopt import __version__
from pyomo.contrib.gdpopt.util import (
from pyomo.core.base import Objective, value, minimize, maximize
from pyomo.core.staleflag import StaleFlagManager
from pyomo.opt import SolverResults
from pyomo.opt import TerminationCondition as tc
from pyomo.util.model_size import build_model_size_report
class _GDPoptAlgorithm:
    CONFIG = ConfigBlock('GDPopt')
    _add_common_configs(CONFIG)

    def __init__(self, **kwds):
        """
        This is a common init method for all the GDPopt algorithms, so that we
        correctly set up the config arguments and initialize the generic parts
        of the algorithm state.
        """
        self.config = self.CONFIG(kwds.pop('options', {}), preserve_implicit=True)
        self.config.set_value(kwds)
        self.LB = float('-inf')
        self.UB = float('inf')
        self.timing = Bunch()
        self.initialization_iteration = 0
        self.iteration = 0
        self.incumbent_boolean_soln = None
        self.incumbent_continuous_soln = None
        self.original_obj = None
        self._dummy_obj = None
        self.original_util_block = None
        self.log_formatter = '{:>9}   {:>15}   {:>11.5f}   {:>11.5f}   {:>8.2%}   {:>7.2f}  {}'

    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass

    def available(self, exception_flag=True):
        """Solver is always available. Though subsolvers may not be, they will
        raise an error when the time comes.
        """
        return True

    def license_is_valid(self):
        return True

    def version(self):
        """Return a 3-tuple describing the solver version."""
        return __version__
    _metasolver = False

    def solve(self, model, **kwds):
        """Solve the model.

        Parameters
        ----------
        model : Block
            the Pyomo model or block to be solved

        """
        alg = kwds.pop('algorithm', None)
        if alg is None:
            alg = kwds.pop('strategy', None)
        if alg is not None:
            raise ValueError("Changing the algorithm in the solve method is not supported for algorithm-specific GDPopt solvers. Either use SolverFactory('gdpopt') or instantiate a solver with the algorithm you want to use.")
        config = self.config(kwds.pop('options', {}), preserve_implicit=True)
        config.set_value(kwds)
        with lower_logger_level_to(config.logger, tee=config.tee):
            self._log_solver_intro_message(config)
            try:
                with time_code(self.timing, 'total', is_main_timer=True):
                    results = self._gather_problem_info_and_solve_non_gdps(model, config)
                    if not results:
                        self._solve_gdp(model, config)
            finally:
                self._get_final_pyomo_results_object()
                self._log_termination_message(config.logger)
                if self.algorithm == 'LBB':
                    config.logger.warning('09/06/22: The GDPopt LBB algorithm currently has known issues. Please use the results with caution and report any bugs!')
                if self.pyomo_results.solver.termination_condition not in {tc.infeasible, tc.unbounded}:
                    self._transfer_incumbent_to_original_model(config.logger)
                self._delete_original_model_util_block()
            return self.pyomo_results

    def _solve_gdp(self, original_model, config):
        raise NotImplementedError('Derived _GDPoptAlgorithms need to implement the _solve_gdp method.')

    def _log_citation(self, config):
        pass

    def _log_solver_intro_message(self, config):
        config.logger.info('Starting GDPopt version %s using %s algorithm' % ('.'.join(map(str, self.version())), self.algorithm))
        os = StringIO()
        config.display(ostream=os)
        config.logger.info(os.getvalue())
        config.logger.info('\n            If you use this software, you may cite the following:\n            - Implementation:\n            Chen, Q; Johnson, ES; Bernal, DE; Valentin, R; Kale, S;\n            Bates, J; Siirola, JD; Grossmann, IE.\n            Pyomo.GDP: an ecosystem for logic based modeling and optimization\n            development.\n            Optimization and Engineering, 2021.\n            '.strip())
        self._log_citation(config)

    def _log_header(self, logger):
        logger.info('=============================================================================================')
        logger.info('{:^9} | {:^15} | {:^11} | {:^11} | {:^8} | {:^7}\n'.format('Iteration', 'Subproblem Type', 'Lower Bound', 'Upper Bound', ' Gap ', 'Time(s)'))

    @property
    def objective_sense(self):
        if hasattr(self, 'pyomo_results'):
            return self.pyomo_results.problem.sense
        else:
            return None

    def _gather_problem_info_and_solve_non_gdps(self, model, config):
        """Solve the model.

        Args:
            model (Block): a Pyomo model or block to be solved

        """
        logger = config.logger
        self._create_pyomo_results_object_with_problem_info(model, config)
        problem = self.pyomo_results.problem
        if problem.number_of_binary_variables == 0 and problem.number_of_integer_variables == 0 and (problem.number_of_disjunctions == 0):
            cont_results = solve_continuous_problem(model, config)
            self.LB = cont_results.problem.lower_bound
            self.UB = cont_results.problem.upper_bound
            return self.pyomo_results
        util_block = self.original_util_block = add_util_block(model)
        add_disjunct_list(util_block)
        add_boolean_variable_lists(util_block)
        add_algebraic_variable_list(util_block)

    def _update_bounds_after_solve(self, subprob_nm, primal=None, dual=None, logger=None):
        primal_improved = self._update_bounds(primal, dual)
        if logger is not None:
            self._log_current_state(logger, subprob_nm, primal_improved)
        return primal_improved

    def _update_bounds(self, primal=None, dual=None, force_update=False):
        """Update bounds correctly depending on objective sense.

        Args:
            primal: bound from solving subproblem with fixed discrete problem
            solution dual: bound from solving discrete problem (relaxation of
                           original problem)
            force_update: flag so this function will set the bound
                          even if it's not an improvement. (Used at termination
                          if the bounds cross.)
        """
        oldLB = self.LB
        oldUB = self.UB
        primal_bound_improved = False
        if self.objective_sense is minimize:
            if primal is not None and (primal < oldUB or force_update):
                self.UB = primal
                primal_bound_improved = primal < oldUB
            if dual is not None and (dual > oldLB or force_update):
                self.LB = dual
        else:
            if primal is not None and (primal > oldLB or force_update):
                self.LB = primal
                primal_bound_improved = primal > oldLB
            if dual is not None and (dual < oldUB or force_update):
                self.UB = dual
        return primal_bound_improved

    def relative_gap(self):
        """Returns current relative optimality gap.

        Note that this gap is not necessarily monotonically decreasing if at
        some point the primal bound changes signs.
        """
        absolute_gap = abs(self.UB - self.LB)
        return absolute_gap / abs(self.primal_bound() + 1e-10)

    def _log_current_state(self, logger, subproblem_type, primal_improved=False):
        star = '*' if primal_improved else ''
        logger.info(self.log_formatter.format(self.iteration, subproblem_type, self.LB, self.UB, self.relative_gap(), get_main_elapsed_time(self.timing), star))

    def _log_termination_message(self, logger):
        logger.info('\nSolved in {} iterations and {:.5f} seconds\nOptimal objective value {:.10f}\nRelative optimality gap {:.5%}'.format(self.iteration, get_main_elapsed_time(self.timing), self.primal_bound(), self.relative_gap()))

    def primal_bound(self):
        if self.objective_sense is minimize:
            return self.UB
        else:
            return self.LB

    def update_incumbent(self, util_block):
        self.incumbent_continuous_soln = [v.value for v in util_block.algebraic_variable_list]
        self.incumbent_boolean_soln = [v.value for v in util_block.transformed_boolean_variable_list]

    def _update_bounds_after_discrete_problem_solve(self, mip_termination, obj_expr, logger):
        if mip_termination is tc.optimal:
            self._update_bounds_after_solve('discrete', dual=value(obj_expr), logger=logger)
        elif mip_termination is tc.infeasible:
            self._update_dual_bound_to_infeasible()
        elif mip_termination is tc.feasible or tc.unbounded:
            pass
        else:
            raise DeveloperError('Unrecognized termination condition %s when updating the dual bound.' % mip_termination)

    def _update_dual_bound_to_infeasible(self):
        if self.objective_sense == minimize:
            self._update_bounds(dual=float('inf'))
        else:
            self._update_bounds(dual=float('-inf'))

    def _update_primal_bound_to_unbounded(self, config):
        if self.objective_sense == minimize:
            self._update_bounds(primal=float('-inf'))
        else:
            self._update_bounds(primal=float('inf'))
        config.logger.info('GDPopt exiting--GDP is unbounded.')
        self.pyomo_results.solver.termination_condition = tc.unbounded

    def _load_infeasible_termination_status(self, config):
        config.logger.info('GDPopt exiting--problem is infeasible.')
        self.pyomo_results.solver.termination_condition = tc.infeasible

    def bounds_converged(self, config):
        if self.pyomo_results.solver.termination_condition == tc.unbounded:
            return True
        elif self.LB + config.bound_tolerance >= self.UB:
            if self.LB == float('inf') and self.UB == float('inf'):
                self._load_infeasible_termination_status(config)
            elif self.LB == float('-inf') and self.UB == float('-inf'):
                self._load_infeasible_termination_status(config)
            else:
                if self.LB + config.bound_tolerance > self.UB:
                    self._update_bounds(dual=self.primal_bound(), force_update=True)
                self._log_current_state(config.logger, '')
                config.logger.info('GDPopt exiting--bounds have converged or crossed.')
                self.pyomo_results.solver.termination_condition = tc.optimal
            return True
        return False

    def reached_iteration_limit(self, config):
        if config.iterlim is not None and self.iteration >= config.iterlim:
            config.logger.info('GDPopt unable to converge bounds within iteration limit of {} iterations.'.format(config.iterlim))
            self.pyomo_results.solver.termination_condition = tc.maxIterations
            return True
        return False

    def reached_time_limit(self, config):
        elapsed = get_main_elapsed_time(self.timing)
        if config.time_limit is not None and elapsed >= config.time_limit:
            config.logger.info('GDPopt exiting--Did not converge bounds before time limit of {} seconds. '.format(config.time_limit))
            self.pyomo_results.solver.termination_condition = tc.maxTimeLimit
            return True
        return False

    def any_termination_criterion_met(self, config):
        return self.bounds_converged(config) or self.reached_iteration_limit(config) or self.reached_time_limit(config)

    def _create_pyomo_results_object_with_problem_info(self, original_model, config):
        """
        Initialize a results object with results.problem information
        """
        results = self.pyomo_results = SolverResults()
        results.solver.name = 'GDPopt %s - %s' % (self.version(), self.algorithm)
        prob = results.problem
        prob.name = original_model.name
        prob.number_of_nonzeros = None
        num_of = build_model_size_report(original_model)
        prob.number_of_constraints = num_of.activated.constraints
        prob.number_of_disjunctions = num_of.activated.disjunctions
        prob.number_of_variables = num_of.activated.variables
        prob.number_of_binary_variables = num_of.activated.binary_variables
        prob.number_of_continuous_variables = num_of.activated.continuous_variables
        prob.number_of_integer_variables = num_of.activated.integer_variables
        config.logger.info('Original model has %s constraints (%s nonlinear) and %s disjunctions, with %s variables, of which %s are binary, %s are integer, and %s are continuous.' % (num_of.activated.constraints, num_of.activated.nonlinear_constraints, num_of.activated.disjunctions, num_of.activated.variables, num_of.activated.binary_variables, num_of.activated.integer_variables, num_of.activated.continuous_variables))
        active_objectives = list(original_model.component_data_objects(ctype=Objective, active=True, descend_into=True))
        number_of_objectives = len(active_objectives)
        if number_of_objectives == 0:
            config.logger.warning('Model has no active objectives. Adding dummy objective.')
            self._dummy_obj = discrete_obj = Objective(expr=1)
            original_model.add_component(unique_component_name(original_model, 'dummy_obj'), discrete_obj)
        elif number_of_objectives > 1:
            raise ValueError('Model has multiple active objectives.')
        else:
            discrete_obj = active_objectives[0]
        prob.sense = minimize if discrete_obj.sense == 1 else maximize
        return results

    def _transfer_incumbent_to_original_model(self, logger):
        StaleFlagManager.mark_all_as_stale(delayed=False)
        if self.incumbent_boolean_soln is None:
            assert self.incumbent_continuous_soln is None
            logger.info('No feasible solutions found.')
            return
        for var, soln in zip(self.original_util_block.algebraic_variable_list, self.incumbent_continuous_soln):
            var.set_value(soln, skip_validation=True)
        for var, soln in zip(self.original_util_block.boolean_variable_list, self.incumbent_boolean_soln):
            if soln is None:
                var.set_value(soln, skip_validation=True)
            elif soln > 0.5:
                var.set_value(True)
            else:
                var.set_value(False)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    def _delete_original_model_util_block(self):
        """For cleaning up after a solve--we want the original model to be
        untouched except for the solution being loaded"""
        blk = self.original_util_block
        if blk is not None:
            blk.parent_block().del_component(blk)
        if self.original_obj is not None:
            self.original_obj.activate()
        if self._dummy_obj is not None:
            self._dummy_obj.parent_block().del_component(self._dummy_obj)
            self._dummy_obj = None

    def _get_final_pyomo_results_object(self):
        """
        Fill in the results.solver information onto the results object
        """
        results = self.pyomo_results
        results.problem.lower_bound = self.LB
        results.problem.upper_bound = self.UB
        results.solver.iterations = self.iteration
        results.solver.timing = self.timing
        results.solver.user_time = self.timing.total
        results.solver.wallclock_time = self.timing.total
        return results