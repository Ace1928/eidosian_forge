import json
import numpy as np
import pandas as pd
from pytest import raises
from scipy import stats
from triad import to_uuid
from tune._utils import assert_close
from tune.concepts.space import (
def test_template_product():
    data = make_template(dict())
    assert [dict()] == list(data.product_grid())
    data = make_template(dict(a=1, b=2))
    assert [dict(a=1, b=2)] == list(data.product_grid())
    data = make_template(dict(a=1, b=Grid(0, 1)))
    assert [dict(a=1, b=0), dict(a=1, b=1)] == list(data.product_grid())
    u = Grid(0, 1)
    data = make_template(dict(a=u, b=1, c=[u], d=Grid(0, 1)))
    assert [dict(a=0, b=1, c=[0], d=0), dict(a=0, b=1, c=[0], d=1), dict(a=1, b=1, c=[1], d=0), dict(a=1, b=1, c=[1], d=1)] == list(data.product_grid())
    data = make_template(dict(a=1, b=Grid(0, 1), c=Rand(0, 1)))
    assert [dict(a=1, b=0, c=Rand(0, 1)), dict(a=1, b=1, c=Rand(0, 1))] == list(data.product_grid())