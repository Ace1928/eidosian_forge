from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.simulator import Simulator
from pyomo.contrib.sensitivity_toolbox.sens import sensitivity_calculation
def _yy01(m, t):
    return sum((m.pp[kk] * m.yy[t, kk] for kk in m.ij)) * m.dyy[t, (0, 1)] == sum((m.pp[kk] * m.yy[t, kk] for kk in m.ij)) * (m.vp[t] * m.yy[t, (0, 0)] - (m.dd[0, 0] + m.omeg) * m.yy[t, (0, 1)]) - m.pp[0, 1] * (1 - m.eps) * sum((m.bb[kk] * m.eta01[kk] * m.pp[kk] * m.yy[t, kk] for kk in m.ij)) * m.yy[t, (0, 1)]