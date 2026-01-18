import json
import numpy as np
import pandas as pd
from pytest import raises
from scipy import stats
from triad import to_uuid
from tune._utils import assert_close
from tune.concepts.space import (
def test_template_misc():
    t = to_template(dict(a=1, b=Grid(0, 1)))
    assert isinstance(t, TuningParametersTemplate)
    t2 = to_template(t)
    assert t is t2
    t3 = to_template(t.encode())
    assert t == t3
    raises(ValueError, lambda: to_template(123))
    u = Grid(0, 1)
    t1 = make_template(dict(a=1, b=u, c=Grid(0, 1)))
    t2 = make_template(dict(a=1, b=u, c=Grid(0, 1)))
    t3 = make_template(dict(a=1, b=u, c=u))
    t4 = make_template(dict(a=1, b=u, c=u))
    assert to_uuid(t1) == to_uuid(t2)
    assert to_uuid(t2) != to_uuid(t3)
    assert to_uuid(t3) == to_uuid(t4)
    u = Grid(0, 1)
    t1 = make_template(dict(a=1, b=u, c=Grid(0, 1), d=FuncParam(lambda x: x + 1, u)))
    raises(ValueError, lambda: t1.simple_value)
    assert [dict(a=1, b=0, c=0, d=1), dict(a=1, b=0, c=1, d=1), dict(a=1, b=1, c=0, d=2), dict(a=1, b=1, c=1, d=2)] == list(t1.product_grid())
    t2 = make_template(dict(a=1, b=2))
    dict(a=1, b=2) == t2.simple_value
    t2 = make_template(dict(a=1, b=FuncParam(lambda x: x + 1, x=2)))
    assert dict(a=1, b=3) == t2.simple_value