import logging
from pyomo.opt import OptSolver, SolverFactory
@SolverFactory.register('py', doc='Direct python solver interfaces')
class pywrapper(OptSolver):
    """Direct python solver interface"""

    def __new__(cls, *args, **kwds):
        mode = kwds.get('solver_io', 'python')
        if mode is None:
            mode = 'python'
        if mode != 'python':
            logging.getLogger('pyomo.solvers').error("Cannot specify IO mode '%s' for direct python solver interface" % mode)
            return None
        if not 'solver' in kwds:
            logging.getLogger('pyomo.solvers').warning('No solver specified for direct python solver interface')
            return None
        kwds['solver_io'] = 'python'
        solver = kwds['solver']
        del kwds['solver']
        return SolverFactory(solver, **kwds)