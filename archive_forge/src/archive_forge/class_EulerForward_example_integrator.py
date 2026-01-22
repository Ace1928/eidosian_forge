import math
import warnings
import numpy as np
from .util import import_
class EulerForward_example_integrator:
    with_jacobian = False
    integrate_adaptive = None

    @staticmethod
    def integrate_predefined(rhs, jac, y0, xout, **kwargs):
        if kwargs:
            warnings.warn('Ignoring keyword-argumtents: %s' % ', '.join(kwargs.keys()))
        x_old = xout[0]
        yout = [y0[:]]
        f = np.empty(len(y0))
        for i, x in enumerate(xout[1:], 1):
            y = yout[-1]
            h = x - x_old
            rhs(x_old, y, f)
            yout.append(y + h * f)
            x_old = x
        return (np.array(yout), {'nfev': len(xout) - 1})