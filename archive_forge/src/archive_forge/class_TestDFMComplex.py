import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from scipy.linalg import cho_solve_banded
from statsmodels import datasets
from statsmodels.tsa.statespace import (sarimax, structural, dynamic_factor,
class TestDFMComplex(CheckPosteriorMoments):

    @classmethod
    def setup_class(cls, missing=None, *args, **kwargs):
        kwargs['k_factors'] = 1
        kwargs['factor_order'] = 1
        super().setup_class(dynamic_factor.DynamicFactor, *args, missing=missing, use_complex=True, **kwargs)