from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.simulator import Simulator
from pyomo.contrib.sensitivity_toolbox.sens import sensitivity_calculation
def _initDistConst(m):
    return m.phi0 * m.Y0 / sum((m.dd_inv[kk] for kk in m.ij))