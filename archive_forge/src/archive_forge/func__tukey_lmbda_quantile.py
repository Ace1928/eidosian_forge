import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
import scipy.special as sp
from scipy.special._testutils import (
from scipy.special._mptestutils import (
def _tukey_lmbda_quantile(p, lmbda):
    return (p ** lmbda - (1 - p) ** lmbda) / lmbda