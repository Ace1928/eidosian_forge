import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.contrib.doe import ModelOptionLib

            The algebraic equation for mole balance
            z: m.pert
            t: time
            