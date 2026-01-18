import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_equal, assert_allclose, assert_
from scipy.sparse.linalg._isolve import minres
from pytest import raises as assert_raises
def test_x0_is_used_by():
    A, b = get_sample_problem()
    np.random.seed(12345)
    x0 = np.random.rand(10)
    trace = []

    def trace_iterates(xk):
        trace.append(xk)
    minres(A, b, x0=x0, callback=trace_iterates)
    trace_with_x0 = trace
    trace = []
    minres(A, b, callback=trace_iterates)
    assert_(not np.array_equal(trace_with_x0[0], trace[0]))