import numpy as np
import math
import inspect
import scipy.optimize
from scipy._lib._util import check_random_state
def _monte_carlo_step(self):
    """Do one Monte Carlo iteration

        Randomly displace the coordinates, minimize, and decide whether
        or not to accept the new coordinates.
        """
    x_after_step = np.copy(self.x)
    x_after_step = self.step_taking(x_after_step)
    minres = self.minimizer(x_after_step)
    x_after_quench = minres.x
    energy_after_quench = minres.fun
    if not minres.success:
        self.res.minimization_failures += 1
        if self.disp:
            print('warning: basinhopping: local minimization failure')
    if hasattr(minres, 'nfev'):
        self.res.nfev += minres.nfev
    if hasattr(minres, 'njev'):
        self.res.njev += minres.njev
    if hasattr(minres, 'nhev'):
        self.res.nhev += minres.nhev
    accept = True
    for test in self.accept_tests:
        if inspect.signature(test) == _new_accept_test_signature:
            testres = test(res_new=minres, res_old=self.incumbent_minres)
        else:
            testres = test(f_new=energy_after_quench, x_new=x_after_quench, f_old=self.energy, x_old=self.x)
        if testres == 'force accept':
            accept = True
            break
        elif testres is None:
            raise ValueError("accept_tests must return True, False, or 'force accept'")
        elif not testres:
            accept = False
    if hasattr(self.step_taking, 'report'):
        self.step_taking.report(accept, f_new=energy_after_quench, x_new=x_after_quench, f_old=self.energy, x_old=self.x)
    return (accept, minres)