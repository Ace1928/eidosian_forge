import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class TestDataFrameDot(DotSharedTests):

    @pytest.fixture
    def obj(self):
        return DataFrame(np.random.default_rng(2).standard_normal((3, 4)), index=['a', 'b', 'c'], columns=['p', 'q', 'r', 's'])

    @pytest.fixture
    def other(self):
        return DataFrame(np.random.default_rng(2).standard_normal((4, 2)), index=['p', 'q', 'r', 's'], columns=['1', '2'])

    @pytest.fixture
    def expected(self, obj, other):
        return DataFrame(np.dot(obj.values, other.values), index=obj.index, columns=other.columns)

    @classmethod
    def reduced_dim_assert(cls, result, expected):
        """
        Assertion about results with 1 fewer dimension that self.obj
        """
        tm.assert_series_equal(result, expected, check_names=False)
        assert result.name is None