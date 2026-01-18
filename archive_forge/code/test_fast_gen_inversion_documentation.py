import pytest
import warnings
import numpy as np
from numpy.testing import (assert_array_equal, assert_allclose,
from copy import deepcopy
from scipy.stats.sampling import FastGeneratorInversion
from scipy import stats
 pdf, cdf etc should map scalar values to scalars. check with and
    w/o domain since domain impacts pdf, cdf etc
    Take x inside and outside of domain 