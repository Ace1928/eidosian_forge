import os
import sys
import time
from pyomo.common.dependencies import mpi4py
from pyomo.contrib.benders.benders_cuts import BendersCutGenerator
import pyomo.environ as pyo
def LimitAmountSold_rule(m, i):
    return m.QuantitySubQuotaSold[i] + m.QuantitySuperQuotaSold[i] - farmer.crop_yield[scenario][i] * m.devoted_acreage[i] <= 0.0