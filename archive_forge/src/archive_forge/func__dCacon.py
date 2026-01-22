import json
from os.path import join, abspath, dirname
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
def _dCacon(m, t):
    if t == 0:
        return Constraint.Skip
    return m.dCa[t] == m.Fa[t] / m.Vr[t] - m.k1 * exp(-m.E1 / (m.R * m.Tr[t])) * m.Ca[t]