import io
import sys
import logging
import os
import abc
from pyomo.common.deprecation import relocated_module_attribute
from pyomo.common.dependencies import attempt_import, numpy as np, numpy_available
from pyomo.common.tee import redirect_fd, TeeStream
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.common.timing import TicTocTimer
from pyomo.core.base import Block, Objective, minimize
from pyomo.opt import SolverStatus, SolverResults, TerminationCondition, ProblemSense
from pyomo.opt.results.solution import Solution
class PyomoCyIpoptSolver(object):
    CONFIG = ConfigBlock('cyipopt')
    CONFIG.declare('tee', ConfigValue(default=False, domain=bool, description='Stream solver output to console'))
    CONFIG.declare('load_solutions', ConfigValue(default=True, domain=bool, description='Store the final solution into the original Pyomo model'))
    CONFIG.declare('return_nlp', ConfigValue(default=False, domain=bool, description='Return the results object and the underlying nlp NLP object from the solve call.'))
    CONFIG.declare('options', ConfigBlock(implicit=True))
    CONFIG.declare('intermediate_callback', ConfigValue(default=None, description='Set the function that will be called each iteration.'))
    CONFIG.declare('halt_on_evaluation_error', ConfigValue(default=None, description='Whether to halt if a function or derivative evaluation fails'))

    def __init__(self, **kwds):
        """Create an instance of the CyIpoptSolver. You must
        provide a problem_interface that corresponds to
        the abstract class CyIpoptProblemInterface

        options can be provided as a dictionary of key value
        pairs
        """
        self.config = self.CONFIG(kwds)

    def _set_model(self, model):
        self._model = model

    def available(self, exception_flag=False):
        return bool(numpy_available and cyipopt_interface.cyipopt_available)

    def license_is_valid(self):
        return True

    def version(self):
        return tuple((int(_) for _ in cyipopt.__version__.split('.')))

    def solve(self, model, **kwds):
        config = self.config(kwds, preserve_implicit=True)
        if not isinstance(model, Block):
            raise ValueError('PyomoCyIpoptSolver.solve(model): model must be a Pyomo Block')
        grey_box_blocks = list(model.component_data_objects(egb.ExternalGreyBoxBlock, active=True))
        if grey_box_blocks:
            nlp = pyomo_grey_box.PyomoNLPWithGreyBoxBlocks(model)
        else:
            nlp = pyomo_nlp.PyomoNLP(model)
        problem = cyipopt_interface.CyIpoptNLP(nlp, intermediate_callback=config.intermediate_callback, halt_on_evaluation_error=config.halt_on_evaluation_error)
        ng = len(problem.g_lb())
        nx = len(problem.x_lb())
        cyipopt_solver = problem
        obj_scaling, x_scaling, g_scaling = problem.scaling_factors()
        if any((_ is not None for _ in (obj_scaling, x_scaling, g_scaling))):
            if obj_scaling is None:
                obj_scaling = 1.0
            if x_scaling is None:
                x_scaling = np.ones(nx)
            if g_scaling is None:
                g_scaling = np.ones(ng)
            try:
                set_scaling = cyipopt_solver.set_problem_scaling
            except AttributeError:
                set_scaling = cyipopt_solver.setProblemScaling
            set_scaling(obj_scaling, x_scaling, g_scaling)
        try:
            add_option = cyipopt_solver.add_option
        except AttributeError:
            add_option = cyipopt_solver.addOption
        for k, v in config.options.items():
            add_option(k, v)
        timer = TicTocTimer()
        try:
            with TeeStream(sys.stdout) as _teeStream:
                if config.tee:
                    try:
                        fd = sys.stdout.fileno()
                    except (io.UnsupportedOperation, AttributeError):
                        fd = _teeStream.STDOUT.fileno()
                else:
                    fd = None
                with redirect_fd(fd=1, output=fd, synchronize=False):
                    x, info = cyipopt_solver.solve(problem.x_init())
            solverStatus = SolverStatus.ok
        except:
            msg = 'Exception encountered during cyipopt solve:'
            logger.error(msg, exc_info=sys.exc_info())
            solverStatus = SolverStatus.unknown
            raise
        wall_time = timer.toc(None)
        results = SolverResults()
        if config.load_solutions:
            nlp.set_primals(x)
            nlp.set_duals(info['mult_g'])
            nlp.load_state_into_pyomo(bound_multipliers=(info['mult_x_L'], info['mult_x_U']))
        else:
            soln = Solution()
            sm = nlp.symbol_map
            soln.variable.update(((sm.getSymbol(i), {'Value': j, 'ipopt_zL_out': zl, 'ipopt_zU_out': zu}) for i, j, zl, zu in zip(nlp.get_pyomo_variables(), x, info['mult_x_L'], info['mult_x_U'])))
            soln.constraint.update(((sm.getSymbol(i), {'Dual': j}) for i, j in zip(nlp.get_pyomo_constraints(), info['mult_g'])))
            model.solutions.add_symbol_map(sm)
            results._smap_id = id(sm)
            results.solution.insert(soln)
        results.problem.name = model.name
        obj = next(model.component_data_objects(Objective, active=True))
        if obj.sense == minimize:
            results.problem.sense = ProblemSense.minimize
            results.problem.upper_bound = info['obj_val']
        else:
            results.problem.sense = ProblemSense.maximize
            results.problem.lower_bound = info['obj_val']
        results.problem.number_of_objectives = 1
        results.problem.number_of_constraints = ng
        results.problem.number_of_variables = nx
        results.problem.number_of_binary_variables = 0
        results.problem.number_of_integer_variables = 0
        results.problem.number_of_continuous_variables = nx
        results.solver.name = 'cyipopt'
        results.solver.return_code = info['status']
        results.solver.message = info['status_msg']
        results.solver.wallclock_time = wall_time
        status_enum = _cyipopt_status_enum[info['status_msg']]
        results.solver.termination_condition = _ipopt_term_cond[status_enum]
        results.solver.status = TerminationCondition.to_solver_status(results.solver.termination_condition)
        problem.close()
        if config.return_nlp:
            return (results, nlp)
        return results

    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass