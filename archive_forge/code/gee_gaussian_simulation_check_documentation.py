from statsmodels.compat.python import lrange
import scipy
import numpy as np
from itertools import product
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Autoregressive, Nested

Assesment of Generalized Estimating Equations using simulation.

This script checks Gaussian models.

See the generated file "gee_gaussian_simulation_check.txt" for
results.
