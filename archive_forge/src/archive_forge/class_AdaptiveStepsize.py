import numpy as np
import math
import inspect
import scipy.optimize
from scipy._lib._util import check_random_state
class AdaptiveStepsize:
    """
    Class to implement adaptive stepsize.

    This class wraps the step taking class and modifies the stepsize to
    ensure the true acceptance rate is as close as possible to the target.

    Parameters
    ----------
    takestep : callable
        The step taking routine.  Must contain modifiable attribute
        takestep.stepsize
    accept_rate : float, optional
        The target step acceptance rate
    interval : int, optional
        Interval for how often to update the stepsize
    factor : float, optional
        The step size is multiplied or divided by this factor upon each
        update.
    verbose : bool, optional
        Print information about each update

    """

    def __init__(self, takestep, accept_rate=0.5, interval=50, factor=0.9, verbose=True):
        self.takestep = takestep
        self.target_accept_rate = accept_rate
        self.interval = interval
        self.factor = factor
        self.verbose = verbose
        self.nstep = 0
        self.nstep_tot = 0
        self.naccept = 0

    def __call__(self, x):
        return self.take_step(x)

    def _adjust_step_size(self):
        old_stepsize = self.takestep.stepsize
        accept_rate = float(self.naccept) / self.nstep
        if accept_rate > self.target_accept_rate:
            self.takestep.stepsize /= self.factor
        else:
            self.takestep.stepsize *= self.factor
        if self.verbose:
            print('adaptive stepsize: acceptance rate {:f} target {:f} new stepsize {:g} old stepsize {:g}'.format(accept_rate, self.target_accept_rate, self.takestep.stepsize, old_stepsize))

    def take_step(self, x):
        self.nstep += 1
        self.nstep_tot += 1
        if self.nstep % self.interval == 0:
            self._adjust_step_size()
        return self.takestep(x)

    def report(self, accept, **kwargs):
        """called by basinhopping to report the result of the step"""
        if accept:
            self.naccept += 1