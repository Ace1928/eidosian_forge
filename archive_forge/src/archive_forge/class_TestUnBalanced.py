import numpy as np
from numpy.testing import assert_equal, assert_raises
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.grouputils import GroupSorted
class TestUnBalanced(CheckPanelLagMixin):

    @classmethod
    def setup_class(cls):
        cls.gind = gind = np.repeat([0, 1, 2], [3, 5, 10])
        cls.mlag = 10
        x = np.arange(18)
        x += 10 ** gind
        cls.x = x[:, None]
        cls.alle = {0: np.array([[1, 2, 3, 13, 14, 15, 16, 17, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117]]), 1: np.array([[2, 3, 14, 15, 16, 17, 109, 110, 111, 112, 113, 114, 115, 116, 117]]), 2: np.array([[3, 15, 16, 17, 110, 111, 112, 113, 114, 115, 116, 117]]), 3: np.array([[16, 17, 111, 112, 113, 114, 115, 116, 117]]), 4: np.array([[17, 112, 113, 114, 115, 116, 117]]), 5: np.array([[113, 114, 115, 116, 117]])}
        cls.calculate()