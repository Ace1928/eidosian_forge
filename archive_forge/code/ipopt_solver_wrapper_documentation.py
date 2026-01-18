from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import TerminationCondition

    Run the solver (must be ipopt) and return the convergence statistics

    Parameters
    ----------
    model : Pyomo model
       The pyomo model to be solved

    solver : Pyomo solver
       The pyomo solver to use - it must be ipopt, but with whichever options are preferred

    max_iter : int
       The maximum number of iterations to allow for ipopt

    max_cpu_time : int
       The maximum cpu time to allow for ipopt (in seconds)

    Returns
    -------
       Returns a tuple with (solve status object, bool (solve successful or not), number of iters, solve time, regularization value at solution)
    