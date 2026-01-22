import numpy as np
import pandas as pd
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.genmod.families import links
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Independence
from numpy.testing import assert_allclose
class TestCompareGaussian(CheckGEEGLM):

    @classmethod
    def setup_class(cls):
        vs = Independence()
        family = families.Gaussian()
        np.random.seed(987126)
        Y = np.random.normal(size=100)
        X1 = np.random.normal(size=100)
        X2 = np.random.normal(size=100)
        X3 = np.random.normal(size=100)
        groups = np.kron(np.arange(20), np.ones(5))
        D = pd.DataFrame({'Y': Y, 'X1': X1, 'X2': X2, 'X3': X3})
        md = GEE.from_formula('Y ~ X1 + X2 + X3', groups, D, family=family, cov_struct=vs)
        cls.result1 = md.fit()
        cls.result2 = GLM.from_formula('Y ~ X1 + X2 + X3', data=D).fit()