import os
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.tests.results import results_kalman_filter
from statsmodels.tsa.statespace.sarimax import SARIMAX
class TestClark1989Alternate(TestClark1989):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(*args, alternate_timing=True, **kwargs)

    def test_using_alterate(self):
        assert self.model._kalman_filter.filter_timing == 1