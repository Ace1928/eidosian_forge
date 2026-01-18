import io
import os
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import patsy
from statsmodels.api import families
from statsmodels.tools.sm_exceptions import (
from statsmodels.othermod.betareg import BetaModel
from .results import results_betareg as resultsb
def test_resid(self):
    res1 = self.res1
    res2 = self.res2
    assert_allclose(res1.fittedvalues, res2.resid['fittedvalues'], rtol=1e-08)
    assert_allclose(res1.resid, res2.resid['response'], atol=1e-08, rtol=1e-08)