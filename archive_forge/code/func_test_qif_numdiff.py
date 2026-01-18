import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.genmod.qif import (QIF, QIFIndependence, QIFExchangeable,
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.genmod import families
@pytest.mark.parametrize('fam', [families.Gaussian(), families.Poisson(), families.Binomial()])
@pytest.mark.parametrize('cov_struct', [QIFIndependence(), QIFExchangeable(), QIFAutoregressive()])
def test_qif_numdiff(fam, cov_struct):
    np.random.seed(234234)
    n = 200
    q = 4
    x = np.random.normal(size=(n, 3))
    if isinstance(fam, families.Gaussian):
        e = np.kron(np.random.normal(size=n // q), np.ones(q))
        e = np.sqrt(0.5) * e + np.sqrt(1 - 0.5 ** 2) * np.random.normal(size=n)
        y = x.sum(1) + e
    elif isinstance(fam, families.Poisson):
        y = np.random.poisson(5, size=n)
    elif isinstance(fam, families.Binomial):
        y = np.random.randint(0, 2, size=n)
    g = np.kron(np.arange(n // q), np.ones(q)).astype(int)
    model = QIF(y, x, groups=g, family=fam, cov_struct=cov_struct)
    for _ in range(5):
        pt = np.random.normal(size=3)
        _, grad, _, _, gn_deriv = model.objective(pt)

        def llf_gn(params):
            return model.objective(params)[3]
        gn_numdiff = approx_fprime(pt, llf_gn, 1e-07)
        assert_allclose(gn_deriv, gn_numdiff, 0.0001)

        def llf(params):
            return model.objective(params)[0]
        grad_numdiff = approx_fprime(pt, llf, 1e-07)
        assert_allclose(grad, grad_numdiff, 0.0001)