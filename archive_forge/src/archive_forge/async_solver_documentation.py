from pyomo.common import Factory
from pyomo.opt.parallel.manager import AsynchronousActionManager

        A simple utility to apply a solver to a list of problem instances.
        The solver is applied asynchronously and a barrier synchronization
        is performed to finalize all results.  All keywords are passed
        to each invocation of the solver, and the results are loaded
        into each instance.

        The solver manager manages this process, and the solver is used to
        manage each invocation of the solver.
        