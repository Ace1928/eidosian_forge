import math
import warnings
import numpy as np
from .util import import_
class EulerBackward_example_integrator:
    with_jacobian = True
    integrate_adaptive = None

    @staticmethod
    def integrate_predefined(rhs, jac, y0, xout, **kwargs):
        if kwargs:
            warnings.warn('Ignoring keyword-argumtents: %s' % ', '.join(kwargs.keys()))
        x_old = xout[0]
        yout = [y0[:]]
        f = np.empty(len(y0))
        j = np.empty((len(y0), len(y0)))
        I = np.eye(len(y0))
        for i, x in enumerate(xout[1:], 1):
            y = yout[-1]
            h = x - x_old
            jac(x_old, y, j)
            lu_piv = lu_factor(h * j - I)
            rhs(x, y, f)
            ynew = y + f * h
            norm_delta_ynew = float('inf')
            while norm_delta_ynew > 1e-12:
                rhs(x, ynew, f)
                delta_ynew = lu_solve(lu_piv, ynew - y - f * h)
                ynew += delta_ynew
                norm_delta_ynew = np.sqrt(np.sum(np.square(delta_ynew)))
            yout.append(ynew)
            x_old = x
        return (np.array(yout), {'nfev': len(xout) - 1})