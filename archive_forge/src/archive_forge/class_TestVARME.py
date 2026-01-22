import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from scipy.linalg import cho_solve_banded
from statsmodels import datasets
from statsmodels.tsa.statespace import (sarimax, structural, dynamic_factor,
class TestVARME(CheckPosteriorMoments):

    @classmethod
    def setup_class(cls, missing=None, *args, **kwargs):
        kwargs['order'] = (1, 0)
        kwargs['measurement_error'] = True
        super().setup_class(varmax.VARMAX, *args, missing=missing, **kwargs)