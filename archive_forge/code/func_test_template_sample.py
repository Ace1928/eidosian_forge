import json
import numpy as np
import pandas as pd
from pytest import raises
from scipy import stats
from triad import to_uuid
from tune._utils import assert_close
from tune.concepts.space import (
def test_template_sample():
    data = make_template(dict())
    raises(ValueError, lambda: list(data.sample(0, 0)))
    raises(ValueError, lambda: list(data.sample(-1, 0)))
    assert [dict()] == list(data.sample(100, 0))
    data = make_template(dict(a=1, b=2))
    assert [dict(a=1, b=2)] == list(data.sample(100, 0))
    data = make_template(dict(a=1, b=Rand(0, 1)))
    assert list(data.sample(10, 0)) == list(data.sample(10, 0))
    assert list(data.sample(10, 0)) != list(data.sample(10, 1))
    a = list(data.sample(10, 0))
    assert 10 == len(a)
    assert all((x.template['b'] >= 0 and x.template['b'] <= 1 for x in a))
    assert all((x.empty for x in a))
    assert all((not x.has_grid for x in a))
    assert all((not x.has_stochastic for x in a))
    u = Rand(0, 1)
    data = make_template(dict(a=1, b=u, c=Grid(0, 1), d=[u]))
    a = list(data.sample(10, 0))
    assert 10 == len(a)
    assert all((x.template['b'] >= 0 and x.template['b'] <= 1 for x in a))
    assert all((x.template['d'][0] == x.template['b'] for x in a))
    assert all((not x.empty for x in a))
    assert all((x.has_grid for x in a))
    assert all((not x.has_stochastic for x in a))