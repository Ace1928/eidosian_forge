from statsmodels.compat.platform import PLATFORM_OSX
import os
import csv
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
import pytest
from statsmodels.regression.mixed_linear_model import (
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from statsmodels.base import _penalties as penalties
import statsmodels.tools.numdiff as nd
from .results import lme_r_results
def test_summary_xname_fe(self):
    summ = self.res.summary(xname_fe=['Constant', 'Age', 'Weight'])
    desired = ['Constant', 'Age', 'Weight', 'Group Var']
    actual = summ.tables[1].index.values
    assert_equal(actual, desired)