from statsmodels.compat import lrange
import os
import numpy as np
import pytest
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
import statsmodels.genmod.generalized_estimating_equations as gee
import statsmodels.tools as tools
import statsmodels.regression.linear_model as lm
from statsmodels.genmod import families
from statsmodels.genmod import cov_struct
import statsmodels.discrete.discrete_model as discrete
import pandas as pd
from scipy.stats.distributions import norm
import warnings
def test_unstructured_incomplete():
    np.random.seed(43)
    ngrp = 400
    cov = np.asarray([[1, 0.7, 0.2], [0.7, 1, 0.5], [0.2, 0.5, 1]])
    covr = np.linalg.cholesky(cov)
    e = np.random.normal(size=(ngrp, 3))
    e = np.dot(e, covr.T)
    xmat = np.random.normal(size=(3 * ngrp, 3))
    par = np.r_[1, -2, 0.1]
    ey = np.dot(xmat, par)
    yl, xl, tl, gl = ([], [], [], [])
    for i in range(ngrp):
        ix = [0, 1, 2]
        ix.pop(i % 3)
        ix = np.asarray(ix)
        tl.append(ix)
        yl.append(ey[3 * i + ix] + e[i, ix])
        x = xmat[3 * i + ix, :]
        xl.append(x)
        gl.append(i * np.ones(2))
    y = np.concatenate(yl)
    x = np.concatenate(xl, axis=0)
    t = np.concatenate(tl)
    t = np.asarray(t, dtype=int)
    g = np.concatenate(gl)
    m = gee.GEE(y, x, time=t[:, None], cov_struct=cov_struct.Unstructured(), groups=g)
    r = m.fit()
    assert_allclose(r.params, par, 0.05, 0.5)
    assert_allclose(m.cov_struct.dep_params, cov, 0.05, 0.5)