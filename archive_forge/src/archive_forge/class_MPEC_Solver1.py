import time
import pyomo.opt
from pyomo.opt import SolverFactory
from pyomo.core import TransformationFactory
from pyomo.common.collections import Bunch
@SolverFactory.register('mpec_nlp', doc='MPEC solver that optimizes a nonlinear transformation')
class MPEC_Solver1(pyomo.opt.OptSolver):

    def __init__(self, **kwds):
        kwds['type'] = 'mpec_nlp'
        pyomo.opt.OptSolver.__init__(self, **kwds)
        self._metasolver = True

    def _presolve(self, *args, **kwds):
        self._instance = args[0]
        pyomo.opt.OptSolver._presolve(self, *args, **kwds)

    def _apply_solver(self):
        start_time = time.time()
        xfrm = TransformationFactory('mpec.simple_nonlinear')
        xfrm.apply_to(self._instance)
        solver = self.options.solver
        if not self.options.solver:
            self.options.solver = solver = 'ipopt'
        with pyomo.opt.SolverFactory(solver) as opt:
            self.results = []
            epsilon_final = self.options.get('epsilon_final', 1e-07)
            epsilon = self.options.get('epsilon_initial', epsilon_final)
            while True:
                self._instance.mpec_bound.set_value(epsilon)
                res = opt.solve(self._instance, tee=self._tee, timelimit=self._timelimit)
                self.results.append(res)
                epsilon /= 10.0
                if epsilon < epsilon_final:
                    break
            from pyomo.mpec import Complementarity
            for cuid in self._instance._transformation_data['mpec.simple_nonlinear'].compl_cuids:
                cobj = cuid.find_component_on(self._instance)
                cobj.parent_block().reclassify_component_type(cobj, Complementarity)
            stop_time = time.time()
            self.wall_time = stop_time - start_time
            return Bunch(rc=getattr(opt, '_rc', None), log=getattr(opt, '_log', None))

    def _postsolve(self):
        results = pyomo.opt.SolverResults()
        solv = results.solver
        solv.name = self.options.subsolver
        solv.wallclock_time = self.wall_time
        cpu_ = []
        for res in self.results:
            if not getattr(res.solver, 'cpu_time', None) is None:
                cpu_.append(res.solver.cpu_time)
        if len(cpu_) > 0:
            solv.cpu_time = sum(cpu_)
        self._instance.compute_statistics()
        prob = results.problem
        prob.name = self._instance.name
        prob.number_of_constraints = self._instance.statistics.number_of_constraints
        prob.number_of_variables = self._instance.statistics.number_of_variables
        prob.number_of_binary_variables = self._instance.statistics.number_of_binary_variables
        prob.number_of_integer_variables = self._instance.statistics.number_of_integer_variables
        prob.number_of_continuous_variables = self._instance.statistics.number_of_continuous_variables
        prob.number_of_objectives = self._instance.statistics.number_of_objectives
        self._instance.solutions.store_to(results)
        self._instance = None
        return results