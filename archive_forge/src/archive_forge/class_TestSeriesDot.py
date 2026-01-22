import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class TestSeriesDot(DotSharedTests):

    @pytest.fixture
    def obj(self):
        return Series(np.random.default_rng(2).standard_normal(4), index=['p', 'q', 'r', 's'])

    @pytest.fixture
    def other(self):
        return DataFrame(np.random.default_rng(2).standard_normal((3, 4)), index=['1', '2', '3'], columns=['p', 'q', 'r', 's']).T

    @pytest.fixture
    def expected(self, obj, other):
        return Series(np.dot(obj.values, other.values), index=other.columns)

    @classmethod
    def reduced_dim_assert(cls, result, expected):
        """
        Assertion about results with 1 fewer dimension that self.obj
        """
        tm.assert_almost_equal(result, expected)