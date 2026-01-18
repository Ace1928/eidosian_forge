import numpy as np
from scipy import stats
from tune._utils import (
def test_uniform_to_continuous():
    assert 10 == uniform_to_continuous(0.5, 10, 10)
    assert 10 == uniform_to_continuous(0.5, 10, 9)
    assert 12.5 == uniform_to_continuous(0.5, 10, 15)
    assert 10 == uniform_to_continuous(0.0, 10, 15)
    assert 15 == uniform_to_continuous(1.0, 10, 15)
    np.random.seed(0)
    values = np.random.uniform(0, 1.0, 100000)
    res = uniform_to_continuous(values, 10, 15)
    assert (res >= 10).all() and (res < 15).all()
    t = stats.kstest(res, 'uniform', args=(10, 5))
    assert t.pvalue > 0.4
    start, end = (1000, 500000)
    res = uniform_to_continuous(values, start, end, log=True)
    assert (res >= start).all() and (res < end).all()
    t = stats.kstest(np.log(res), 'uniform', args=(np.log(start), np.log(end) - np.log(start)))
    assert t.pvalue > 0.4
    t = stats.kstest(res, 'uniform', args=(start, end - start))
    assert t.pvalue < 0.001
    res = uniform_to_continuous(values, start, end, log=True, base=10)
    assert (res >= start).all() and (res < end).all()
    b = np.log(10)
    t = stats.kstest(np.log(res) / b, 'uniform', args=(np.log(start) / b, np.log(end) / b - np.log(start) / b))
    assert t.pvalue > 0.4
    t = stats.kstest(res, 'uniform', args=(start, end - start))
    assert t.pvalue < 0.001