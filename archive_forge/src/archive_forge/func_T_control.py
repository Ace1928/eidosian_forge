import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.contrib.doe import ModelOptionLib
def T_control(m, t):
    """
            T at interval timepoint equal to the T of the control time point at the beginning of this interval
            Count how many control points are before the current t;
            locate the nearest neighbouring control point before this t
            """
    if t in m.t_con:
        return pyo.Constraint.Skip
    else:
        neighbour_t = max((tc for tc in control_time if tc < t))
        return m.T[t] == m.T[neighbour_t]