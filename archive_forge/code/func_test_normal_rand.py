import json
import numpy as np
import pandas as pd
from pytest import raises
from scipy import stats
from triad import to_uuid
from tune._utils import assert_close
from tune.concepts.space import (
def test_normal_rand():
    with raises(ValueError):
        NormalRand(1.0, 0.0)
    with raises(ValueError):
        NormalRand(1.0, -1.0)
    v = NormalRand(0.05, 0.2)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    res = v.generate_many(100000, 0)
    t = stats.kstest(res, 'norm', args=(0.05, 0.2))
    assert t.pvalue > 0.4
    v = NormalRand(0.05, 0.2, q=0.1)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    actual = [x for x in v.generate_many(1000, 0) if x >= -0.155 and x <= 0.255]
    assert_close([-0.15, -0.05, 0.05, 0.15, 0.25], actual)
    v2 = NormalRand(0.05, 0.2, q=0.1)
    v3 = Rand(0.05, 0.2, q=0.1)
    assert to_uuid(v) == to_uuid(v2)
    assert to_uuid(v) != to_uuid(v3)