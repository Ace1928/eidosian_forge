import math
import logging
from pyomo.common.errors import InfeasibleConstraintException, IntervalException
def _inverse_abs(zl, zu):
    if zl < 0:
        zl = 0
    if zu < 0:
        zu = 0
    xu = max(zl, zu)
    xl = -xu
    return (xl, xu)