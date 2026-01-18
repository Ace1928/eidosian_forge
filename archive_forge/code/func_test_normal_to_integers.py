import numpy as np
from scipy import stats
from tune._utils import (
def test_normal_to_integers():
    assert 17 == normal_to_integers(0, 17, 19)
    np.random.seed(0)
    values = np.random.normal(0, 1.0, 1000)
    res = normal_to_integers(values, 17, 19)
    assert 17 in res
    res = normal_to_integers(values, 17, 19, q=3)
    res = [x for x in res if x >= 10 and x <= 26]
    assert_close([11, 14, 17, 20, 23, 26], res)