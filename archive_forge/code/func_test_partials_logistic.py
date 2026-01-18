import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tsa.regime_switching import markov_switching
def test_partials_logistic():
    logistic = markov_switching._logistic
    partials_logistic = markov_switching._partials_logistic
    cases = [0, 10.0, -4]
    for x in cases:
        assert_allclose(partials_logistic(x), logistic(x) - logistic(x) ** 2)
        assert_allclose(partials_logistic(x), approx_fprime_cs([x], logistic))
    cases = [[1.0], [0, 1.0], [-2, 3.0, 1.2, -30.0]]
    for x in cases:
        evaluated = np.atleast_1d(logistic(x))
        partials = np.diag(evaluated - evaluated ** 2)
        for i in range(len(x)):
            for j in range(i):
                partials[i, j] = partials[j, i] = -evaluated[i] * evaluated[j]
        assert_allclose(partials_logistic(x), partials)
        assert_allclose(partials_logistic(x), approx_fprime_cs(x, logistic))
    case = [[1.0]]
    evaluated = logistic(case)
    partial = [evaluated - evaluated ** 2]
    assert_allclose(partials_logistic(case), partial)
    assert_allclose(partials_logistic(case), approx_fprime_cs(case, logistic))
    case = [[0], [1.0]]
    evaluated = logistic(case)[:, 0]
    partials = np.diag(evaluated - evaluated ** 2)
    partials[0, 1] = partials[1, 0] = -np.multiply(*evaluated)
    assert_allclose(partials_logistic(case)[:, :, 0], partials)
    assert_allclose(partials_logistic(case), approx_fprime_cs(np.squeeze(case), logistic)[..., None])
    case = [[0, 1.0]]
    evaluated = logistic(case)
    partials = (evaluated - evaluated ** 2)[None, ...]
    assert_allclose(partials_logistic(case), partials)
    assert_allclose(partials_logistic(case), approx_fprime_cs(case, logistic).T)
    case = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    evaluated = logistic(case)
    partials = partials_logistic(case)
    for t in range(4):
        for j in range(3):
            desired = np.diag(evaluated[:, j, t] - evaluated[:, j, t] ** 2)
            desired[0, 1] = desired[1, 0] = -np.multiply(*evaluated[:, j, t])
            assert_allclose(partials[..., j, t], desired)