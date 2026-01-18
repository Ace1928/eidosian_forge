import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from scipy import stats
import statsmodels.api as sm
from statsmodels.miscmodels.count import PoissonGMLE, PoissonOffsetGMLE, \
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.tools.sm_exceptions import ValueWarning
Testing GenericLikelihoodModel variations on Poisson


