import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.common.dependencies.matplotlib import pyplot as plt
def _flow_eqn_rule(m, t):
    return m.flow_in[t] - m.flow_out[t] == 0