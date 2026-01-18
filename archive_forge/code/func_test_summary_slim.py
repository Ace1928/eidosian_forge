from statsmodels.compat.python import lrange
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy.linalg import toeplitz
from scipy.stats import t as student_t
from statsmodels.datasets import longley
from statsmodels.regression.linear_model import (
from statsmodels.tools.tools import add_constant
def test_summary_slim(self):
    with warnings.catch_warnings():
        msg = 'kurtosistest only valid for n>=20'
        warnings.filterwarnings('ignore', message=msg, category=UserWarning)
        summ = self.res1.summary(slim=True)
    assert len(summ.tables) == 2
    assert len(str(summ)) < 6700