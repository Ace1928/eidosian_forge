import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state
from scipy.sparse import (csr_matrix, coo_matrix,
from scipy.sparse._construct import rand as sprand
def test_randn(self):
    random_states = [None, 4321, np.random.RandomState()]
    try:
        gen = np.random.default_rng()
        random_states.append(gen)
    except AttributeError:
        pass
    for rs in random_states:
        x = _sprandn(10, 20, density=0.5, dtype=np.float64, random_state=rs)
        assert_(np.any(np.less(x.data, 0)))
        assert_(np.any(np.less(1, x.data)))
        x = _sprandn_array(10, 20, density=0.5, dtype=np.float64, random_state=rs)
        assert_(np.any(np.less(x.data, 0)))
        assert_(np.any(np.less(1, x.data)))