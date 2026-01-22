import logging
from pyomo.core.base import Constraint, Param, value, Suffix, Block
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.dependencies import (
def _simulate_with_casadi_with_inputs(self, initcon, tsim, varying_inputs, integrator, integrator_options):
    xalltemp = [self._templatemap[i] for i in self._diffvars]
    xall = casadi.vertcat(*xalltemp)
    time = casadi.SX.sym('time')
    odealltemp = [time * convert_pyomo2casadi(self._rhsdict[i]) for i in self._derivlist]
    odeall = casadi.vertcat(*odealltemp)
    ptemp = [self._templatemap[i] for i in self._siminputvars.values()]
    pall = casadi.vertcat(time, *ptemp)
    dae = {'x': xall, 'p': pall, 'ode': odeall}
    if len(self._algvars) != 0:
        zalltemp = [self._templatemap[i] for i in self._simalgvars]
        zall = casadi.vertcat(*zalltemp)
        algalltemp = [convert_pyomo2casadi(i) for i in self._alglist]
        algall = casadi.vertcat(*algalltemp)
        dae['z'] = zall
        dae['alg'] = algall
    integrator_options['tf'] = 1.0
    F = casadi.integrator('F', integrator, dae, integrator_options)
    N = len(tsim)
    tsimtemp = np.hstack([0, tsim[1:] - tsim[0:-1]])
    tsimtemp.shape = (1, len(tsimtemp))
    palltemp = [casadi.DM(tsimtemp)]
    for p in self._siminputvars.keys():
        profile = varying_inputs[p]
        tswitch = list(profile.keys())
        tswitch.sort()
        tidx = [tsim.searchsorted(i) for i in tswitch] + [len(tsim) - 1]
        ptemp = [profile[0]] + [casadi.repmat(profile[tswitch[i]], 1, tidx[i + 1] - tidx[i]) for i in range(len(tswitch))]
        temp = casadi.horzcat(*ptemp)
        palltemp.append(temp)
    I = F.mapaccum('simulator', N)
    sol = I(x0=initcon, p=casadi.vertcat(*palltemp))
    profile = sol['xf'].full().T
    if len(self._algvars) != 0:
        algprofile = sol['zf'].full().T
        profile = np.concatenate((profile, algprofile), axis=1)
    return [tsim, profile]