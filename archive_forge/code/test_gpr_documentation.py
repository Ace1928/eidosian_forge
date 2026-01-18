import re
import sys
import warnings
import numpy as np
import pytest
from scipy.optimize import approx_fprime
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
from sklearn.gaussian_process.kernels import (
from sklearn.gaussian_process.tests._mini_sequence_kernel import MiniSeqKernel
from sklearn.utils._testing import (

    Check that the input X is not modified by the predict method of the
    GaussianProcessRegressor when setting return_std=True.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/24340
    