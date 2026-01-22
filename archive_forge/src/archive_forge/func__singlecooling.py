import json
from os.path import join, abspath, dirname
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
def _singlecooling(m, t):
    return m.Tc[t] == m.Tj[t]