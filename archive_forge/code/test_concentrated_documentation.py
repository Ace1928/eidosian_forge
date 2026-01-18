import numpy as np
import pandas as pd
from statsmodels.tools.tools import Bunch
from .results import results_varmax
from statsmodels.tsa.statespace import sarimax, varmax
from numpy.testing import assert_raises, assert_allclose

Tests for concentrating the scale out of the loglikelihood function

Note: the univariate cases is well tested in test_sarimax.py

Author: Chad Fulton
License: Simplified-BSD
