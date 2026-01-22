import os
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, structural, varmax
from statsmodels.tsa.statespace.simulation_smoother import (
class TestMultivariateVARKnown(MultivariateVARKnown):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class()
        cls.true_llf = 39.01246166