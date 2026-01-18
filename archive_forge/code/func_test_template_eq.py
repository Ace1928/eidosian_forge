import json
import numpy as np
import pandas as pd
from pytest import raises
from scipy import stats
from triad import to_uuid
from tune._utils import assert_close
from tune.concepts.space import (
def test_template_eq():
    data1 = make_template(dict())
    data2 = make_template(dict())
    assert data1 == data2
    data1 = make_template(dict(a=1, b=2))
    data2 = make_template(dict(a=1, b=2))
    data3 = make_template(dict(a=1, b=3))
    assert data1 == data2
    assert data1 != data3
    data1 = make_template(dict(a=1, b=Grid(0, 1)))
    data2 = make_template(dict(a=1, b=Grid(0, 1)))
    data3 = make_template(dict(a=1, b=Grid(0, 2)))
    assert data1 == data2
    assert data1 != data3
    u = Grid(0, 1)
    v = Grid(0, 1)
    data1 = make_template(dict(a=1, b=u, c=u))
    data2 = dict(a=1, b=v, c=v)
    data3 = dict(a=1, b=u, c=v)
    assert data1 == data2
    assert data1 != data3
    assert data2 == data1
    assert data3 != data1