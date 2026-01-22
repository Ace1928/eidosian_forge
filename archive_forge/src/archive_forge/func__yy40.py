from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.simulator import Simulator
from pyomo.contrib.sensitivity_toolbox.sens import sensitivity_calculation
def _yy40(m, t):
    return m.dyy[t, (4, 0)] == m.dd[3, 0] * m.yy[t, (3, 0)] - (m.dd[4, 0] + m.dd[0, 0]) * m.yy[t, (4, 0)]