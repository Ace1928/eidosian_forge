import json
import numpy as np
import pandas as pd
from pytest import raises
from scipy import stats
from triad import to_uuid
from tune._utils import assert_close
from tune.concepts.space import (
def test_transition_choice():
    raises(ValueError, lambda: TransitionChoice())
    v = TransitionChoice('a', 'b', 'c')
    assert v.generate(0) == v.generate(0)
    assert v.generate(0) != v.generate(1)
    assert v.generate_many(20, 0) == v.generate_many(20, 0)
    assert v.generate_many(20, 0) != v.generate_many(20, 1)
    actual = set(v.generate_many(20, 0))
    assert set(['a', 'b', 'c']) == actual
    assert to_uuid(v) != to_uuid(Grid('a', 'b', 'c'))
    assert v != Grid('a', 'b', 'c')
    v = TransitionChoice(1, 2, 3)
    assert json.loads(json.dumps({'x': v.generate(0)}))['x'] <= 3
    v = TransitionChoice('a', 'b', 'c')
    assert isinstance(json.loads(json.dumps({'x': v.generate(0)}))['x'], str)
    v2 = TransitionChoice('a', 'b', 'c')
    v3 = Choice('a', 'b', 'c')
    assert to_uuid(v) == to_uuid(v2)
    assert to_uuid(v2) != to_uuid(v3)