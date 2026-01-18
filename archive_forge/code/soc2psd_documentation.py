import numpy as np
from scipy import sparse
import cvxpy as cp
from cvxpy import problems
from cvxpy.constraints.psd import PSD
from cvxpy.constraints.second_order import SOC
from cvxpy.reductions.reduction import Reduction

                Constrain M to the PSD cone.
                