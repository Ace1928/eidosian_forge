import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root

        Results above from R cor.test, e.g.

        options(digits=16)
        x <- c(1.76405235, 0.40015721, 0.97873798,
               2.2408932, 1.86755799, -0.97727788)
        y <- c(2.71414076, 0.2488, 0.87551913,
               2.6514917, 2.01160156, 0.47699563)
        cor.test(x, y, method = "spearm", alternative = "t")
        