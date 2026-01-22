import logging
from pyomo.core.base import Constraint, Param, value, Suffix, Block
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.dependencies import (
def _simulate_with_scipy(self, initcon, tsim, switchpts, varying_inputs, integrator, integrator_options):
    scipyint = scipy.integrate.ode(self._rhsfun).set_integrator(integrator, **integrator_options)
    scipyint.set_initial_value(initcon, tsim[0])
    profile = np.array(initcon)
    i = 1
    while scipyint.successful() and scipyint.t < tsim[-1]:
        if tsim[i - 1] in switchpts:
            for v in self._siminputvars.keys():
                if tsim[i - 1] in varying_inputs[v]:
                    p = self._templatemap[self._siminputvars[v]]
                    p.set_value(varying_inputs[v][tsim[i - 1]])
        profilestep = scipyint.integrate(tsim[i])
        profile = np.vstack([profile, profilestep])
        i += 1
    if not scipyint.successful():
        raise DAE_Error('The Scipy integrator %s did not terminate successfully.' % integrator)
    return [tsim, profile]