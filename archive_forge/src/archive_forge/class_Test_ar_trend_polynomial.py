import os
import warnings
from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace import sarimax, tools
from .results import results_sarimax
from statsmodels.tools import add_constant
from statsmodels.tools.tools import Bunch
from numpy.testing import (
class Test_ar_trend_polynomial(SARIMAXCoverageTest):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 0, 0)
        kwargs['trend'] = [1, 0, 0, 1]
        super().setup_class(3, *args, **kwargs)
        tps = cls.true_params
        cls.true_params[:2] = (1 - tps[2:5].sum()) * tps[:2]