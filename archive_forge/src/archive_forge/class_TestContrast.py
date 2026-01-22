import numpy as np
import numpy.random
from numpy.testing import assert_almost_equal, assert_equal
from statsmodels.stats.contrast import Contrast
import statsmodels.stats.contrast as smc
class TestContrast:

    @classmethod
    def setup_class(cls):
        numpy.random.seed(54321)
        cls.X = numpy.random.standard_normal((40, 10))

    def test_contrast1(self):
        term = np.column_stack((self.X[:, 0], self.X[:, 2]))
        c = Contrast(term, self.X)
        test_contrast = [[1] + [0] * 9, [0] * 2 + [1] + [0] * 7]
        assert_almost_equal(test_contrast, c.contrast_matrix)

    def test_contrast2(self):
        zero = np.zeros((40,))
        term = np.column_stack((zero, self.X[:, 2]))
        c = Contrast(term, self.X)
        test_contrast = [0] * 2 + [1] + [0] * 7
        assert_almost_equal(test_contrast, c.contrast_matrix)

    def test_contrast3(self):
        P = np.dot(self.X, np.linalg.pinv(self.X))
        resid = np.identity(40) - P
        noise = np.dot(resid, numpy.random.standard_normal((40, 5)))
        term = np.column_stack((noise, self.X[:, 2]))
        c = Contrast(term, self.X)
        assert_equal(c.contrast_matrix.shape, (10,))

    def test_estimable(self):
        X2 = np.column_stack((self.X, self.X[:, 5]))
        c = Contrast(self.X[:, 5], X2)