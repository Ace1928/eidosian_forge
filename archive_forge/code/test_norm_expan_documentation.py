import pytest
import numpy as np
from scipy import stats
from numpy.testing import assert_allclose, assert_array_less
from statsmodels.sandbox.distributions.extras import NormExpan_gen
Unit tests for Gram-Charlier exansion

No reference results, test based on consistency and normal case.

Created on Wed Feb 19 12:39:49 2014

Author: Josef Perktold
