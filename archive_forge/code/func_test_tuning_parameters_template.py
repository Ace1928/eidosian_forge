import json
import numpy as np
import pandas as pd
from pytest import raises
from scipy import stats
from triad import to_uuid
from tune._utils import assert_close
from tune.concepts.space import (
def test_tuning_parameters_template():
    data = dict(a=1)
    e = make_template(data)
    assert e.empty
    assert not e.has_grid
    assert not e.has_stochastic
    data = dict(a=Rand(0, 1))
    e = make_template(data)
    assert not e.empty
    assert not e.has_grid
    assert e.has_stochastic
    data = dict(a=Grid(0, 1))
    e = make_template(data)
    assert not e.empty
    assert e.has_grid
    assert not e.has_stochastic
    data = dict(a=Rand(0, 1), b=Grid(2, 3), c=dict(a=Rand(10, 20), b=[dict(x=Rand(100, 200))], c=[1, Rand(1000, 2000)], d=None), d=None)
    e = make_template(data)
    assert not e.empty
    assert e.has_grid
    assert e.has_stochastic
    assert [Rand(0, 1), Grid(2, 3), Rand(10, 20), Rand(100, 200), Rand(1000, 2000)] == e.params
    res = e.fill([0.5, 2, 10.5, 100.5, 1000.5])
    res2 = e.fill([0.55, 2, 10.55, 100.5, 1000.5])
    assert dict(a=0.5, b=2, c=dict(a=10.5, b=[dict(x=100.5)], c=[1, 1000.5], d=None), d=None) == res
    assert res2 is not res
    assert dict(a=0.55, b=2, c=dict(a=10.55, b=[dict(x=100.5)], c=[1, 1000.5], d=None), d=None) == res2
    data = dict(a=Rand(0, 1), b=dict(x=[Grid(2, 3)]))
    e = make_template(data)
    assert dict(p0=Rand(0, 1), p1=Grid(2, 3)) == e.params_dict
    assert dict(a=0.5, b=dict(x=[2])) == e.fill_dict(dict(p1=2, p0=0.5))
    expr = Rand(0, 1)
    data = dict(a=expr, b=dict(x=expr), c=Rand(2, 4))
    e = make_template(data)
    assert dict(p0=Rand(0, 1), p1=Rand(2, 4)) == e.params_dict
    assert dict(a=0.5, b=dict(x=0.5), c=2) == e.fill_dict(dict(p1=2, p0=0.5))
    e = make_template(dict(a=Rand(0, 1), b=pd.DataFrame([[0]])))

    def tf(*args, x):
        return sum(args) + x[0]
    u = Grid(0, 1)
    e = make_template(dict(a=1, b=[FuncParam(tf, Rand(0, 1), u, x=[u])]))
    assert e.has_grid
    assert e.has_stochastic
    assert dict(a=1, b=[2.5]) == e.fill([0.5, 1])