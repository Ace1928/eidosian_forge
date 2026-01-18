import warnings
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.datasets import macrodata
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tsa.statespace import structural
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.tests.results import results_structural
def test_random_walk_with_drift(close_figures):
    run_ucm('random_walk_with_drift')
    run_ucm('random_walk_with_drift', use_exact_diffuse=True)