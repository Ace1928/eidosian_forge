import numpy as np
from scipy import stats
from tune._utils import (
def test_normal_to_discrete():
    np.random.seed(0)
    values = np.random.normal(0, 1.0, 100000)
    res = normal_to_discrete(values, 0.3, 3.0, 0.0001)
    t = stats.kstest(res, 'norm', args=(0.3, 3.0))
    assert t.pvalue > 0.4
    res = normal_to_discrete(values, 0.1, 0.9, q=0.3)
    res = [x for x in res if x >= -0.9 and x <= 0.9]
    assert_close([-0.8, -0.5, -0.2, 0.1, 0.4, 0.7], res)