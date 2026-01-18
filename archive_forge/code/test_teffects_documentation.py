import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.discrete.discrete_model import Probit
from statsmodels.treatment.treatment_effects import (
from .results import results_teffects as res_st

Created on Feb 3, 2022 1:04:22 PM

Author: Josef Perktold
License: BSD-3
