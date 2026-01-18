import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.opt import check_optimal_termination
from pyomo.common.dependencies import attempt_import
from pyomo.common.dependencies import pandas as pd, pandas_available
from pyomo.contrib.interior_point.inverse_reduced_hessian import (
there is a binding constraint