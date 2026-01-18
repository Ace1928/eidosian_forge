import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from scipy.stats.contingency import crosstab
def test_validation_at_least_one():
    with pytest.raises(TypeError, match='At least one'):
        crosstab()