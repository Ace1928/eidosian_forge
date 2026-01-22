import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
class TestHoltWintersNoTrendHeuristicInitialization(CheckHeuristicInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(aust, seasonal=4, initialization_method='heuristic')
        super().setup_class(mod)

    def test_heuristic(self):
        TestHoltWintersHeuristicInitialization.test_heuristic(self)