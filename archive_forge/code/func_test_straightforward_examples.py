import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial import geometric_slerp
@pytest.mark.parametrize('start, end, expected', [(np.array([1, 0]), np.array([0, 1]), np.array([[1, 0], [np.sqrt(3) / 2, 0.5], [0.5, np.sqrt(3) / 2], [0, 1]])), (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([[1, 0, 0], [np.sqrt(3) / 2, 0.5, 0], [0.5, np.sqrt(3) / 2, 0], [0, 1, 0]])), (np.array([1, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0]), np.array([[1, 0, 0, 0, 0], [np.sqrt(3) / 2, 0.5, 0, 0, 0], [0.5, np.sqrt(3) / 2, 0, 0, 0], [0, 1, 0, 0, 0]]))])
def test_straightforward_examples(self, start, end, expected):
    actual = geometric_slerp(start=start, end=end, t=np.linspace(0, 1, 4))
    assert_allclose(actual, expected, atol=1e-16)