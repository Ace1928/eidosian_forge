import os
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, structural, varmax
from statsmodels.tsa.statespace.simulation_smoother import (
class TestMultivariateVAR(MultivariateVAR):

    @classmethod
    def setup_class(cls):
        super().setup_class()
        path = os.path.join(current_path, 'results', 'results_simulation_smoothing3_variates.csv')
        cls.variates = pd.read_csv(path).values.squeeze()
        path = os.path.join(current_path, 'results', 'results_simulation_smoothing3.csv')
        cls.true = pd.read_csv(path)
        cls.true_llf = 1695.34872