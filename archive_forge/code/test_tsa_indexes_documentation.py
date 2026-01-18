from statsmodels.compat.pandas import PD_LT_2_2_0, YEAR_END, is_int_index
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base import tsa_model

Test index support in time series models

1. Test support for passing / constructing the underlying index in __init__
2. Test wrapping of output using the underlying index
3. Test wrapping of prediction / forecasting using the underlying index or
   extensions of it.

Author: Chad Fulton
License: BSD-3
