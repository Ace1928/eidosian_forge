import json
import numpy as np
import pandas as pd
from pytest import raises
from scipy import stats
from triad import to_uuid
from tune._utils import assert_close
from tune.concepts.space import (
def test_normal_randint():
    v = NormalRandInt(5, 2)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    actual = set(v.generate_many(50, 0))
    for x in [3, 4, 5, 6, 7]:
        assert x in actual
    v = NormalRandInt(5, 2, q=3)
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(2)
    actual = set(v.generate_many(50, 0))
    for x in [-1, 2, 5, 8, 11]:
        assert x in actual
    assert 6 not in actual
    v2 = NormalRandInt(5, 2, q=3)
    v3 = NormalRand(5, 2, q=3)
    assert to_uuid(v) == to_uuid(v2)
    assert to_uuid(v) != to_uuid(v3)