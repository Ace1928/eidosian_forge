from itertools import product
import numpy as np
import random
import functools
import pytest
from numpy.testing import (assert_, assert_equal, assert_allclose,
from pytest import raises as assert_raises
import scipy.stats as stats
from scipy.stats import distributions
from scipy.stats._hypotests import (epps_singleton_2samp, cramervonmises,
from scipy.stats._mannwhitneyu import mannwhitneyu, _mwu_state
from .common_tests import check_named_results
from scipy._lib._testutils import _TestPythranFunc
@pytest.mark.parametrize('c1, n1, c2, n2, p_expect', ([0, 100, 3, 100, 0.0884], [2, 100, 6, 100, 0.1749]))
def test_paper_examples(self, c1, n1, c2, n2, p_expect):
    res = stats.poisson_means_test(c1, n1, c2, n2)
    assert_allclose(res.pvalue, p_expect, atol=0.0001)