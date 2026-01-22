import os
import sys
import time
from pyomo.common.dependencies import mpi4py
from pyomo.contrib.benders.benders_cuts import BendersCutGenerator
import pyomo.environ as pyo
def EnforceQuotas_rule(m, i):
    return (0.0, m.QuantitySubQuotaSold[i], farmer.PriceQuota[i])