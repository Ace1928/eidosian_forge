import os
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from statsmodels import datasets
from statsmodels.tools import add_constant
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.statespace import mlemodel
def update_direct(self, obs_cov, state_cov_diag):
    self['obs_cov'] = obs_cov
    self[self._state_cov_ix] = state_cov_diag