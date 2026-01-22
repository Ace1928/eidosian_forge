import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.contrib.doe import ModelOptionLib
def alge(m, t):
    """
            The algebraic equation for mole balance
            z: m.pert
            t: time
            """
    return m.C['CA', t] + m.C['CB', t] + m.C['CC', t] == m.CA0[0]