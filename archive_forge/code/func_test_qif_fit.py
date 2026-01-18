import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.genmod.qif import (QIF, QIFIndependence, QIFExchangeable,
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.genmod import families
@pytest.mark.parametrize('fam', [families.Gaussian(), families.Poisson(), families.Binomial()])
@pytest.mark.parametrize('cov_struct', [QIFIndependence(), QIFExchangeable(), QIFAutoregressive()])
def test_qif_fit(fam, cov_struct):
    np.random.seed(234234)
    n = 1000
    q = 4
    params = np.r_[1, -0.5, 0.2]
    x = np.random.normal(size=(n, len(params)))
    if isinstance(fam, families.Gaussian):
        e = np.kron(np.random.normal(size=n // q), np.ones(q))
        e = np.sqrt(0.5) * e + np.sqrt(1 - 0.5 ** 2) * np.random.normal(size=n)
        y = np.dot(x, params) + e
    elif isinstance(fam, families.Poisson):
        lpr = np.dot(x, params)
        mean = np.exp(lpr)
        y = np.random.poisson(mean)
    elif isinstance(fam, families.Binomial):
        lpr = np.dot(x, params)
        mean = 1 / (1 + np.exp(-lpr))
        y = (np.random.uniform(0, 1, size=n) < mean).astype(int)
    g = np.kron(np.arange(n // q), np.ones(q)).astype(int)
    model = QIF(y, x, groups=g, family=fam, cov_struct=cov_struct)
    rslt = model.fit()
    assert_allclose(rslt.params, params, atol=0.05, rtol=0.05)
    _ = rslt.summary()