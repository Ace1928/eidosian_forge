from statsmodels.compat.pandas import QUARTER_END
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tsa.statespace import (

Monte Carlo-type tests for the BM model

Note that that the actual tests that run are just regression tests against
previously estimated values with small sample sizes that can be run quickly
for continuous integration. However, this file can be used to re-run (slow)
large-sample Monte Carlo tests.
