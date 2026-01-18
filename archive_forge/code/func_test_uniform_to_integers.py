import numpy as np
from scipy import stats
from tune._utils import (
def test_uniform_to_integers():
    assert 10 == uniform_to_integers(0.5, 10, 10)
    np.random.seed(0)
    values = np.random.uniform(0, 1.0, 1000)
    res = uniform_to_integers(values, 1, 5)
    assert all((isinstance(x, int) for x in res))
    assert set(res) == set([1, 2, 3, 4, 5])
    assert sum((1 if v == 5 else 0 for v in res)) >= 180
    res = uniform_to_integers(values, 1, 5, include_high=False)
    assert all((isinstance(x, int) for x in res))
    assert set(res) == set([1, 2, 3, 4])
    assert sum((1 if v == 4 else 0 for v in res)) >= 230
    res = uniform_to_integers(values, 1, 5, q=2)
    assert set(res) == set([1, 3, 5])
    assert sum((1 if v == 5 else 0 for v in res)) >= 300
    res = uniform_to_integers(values, 1, 5, q=2, include_high=False)
    assert set(res) == set([1, 3])
    assert sum((1 if v == 3 else 0 for v in res)) >= 480
    for ih in [True, False]:
        res = uniform_to_integers(values, 1, 5, q=3, include_high=ih)
        assert set(res) == set([1, 4])
        assert sum((1 if v == 4 else 0 for v in res)) >= 480