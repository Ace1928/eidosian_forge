import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tsa.regime_switching import markov_switching

General tests for Markov switching models

Author: Chad Fulton
License: BSD-3
