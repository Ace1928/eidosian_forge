import os
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import pandas as pd
import pytest
from statsmodels.sandbox.nonparametric import kernels
class TestEpan(CheckKernelMixin):
    kern_name = 'epan2'
    kern = kernels.Epanechnikov()