from tune import Choice, Grid, Rand, RandInt, Space, TuningParametersTemplate, FuncParam
from pytest import raises
def test_single_space():
    raises(ValueError, lambda: Space('abc'))
    raises(ValueError, lambda: Space(1))
    raises(ValueError, lambda: Space(1, 2))
    space = Space(a=1, b=Grid(2, 3, 4))
    dicts = list(space)
    dicts = list(space)
    assert 3 == len(dicts)
    assert dict(a=1, b=2) == dicts[0]
    assert dict(a=1, b=3) == dicts[1]
    dicts = list(Space(dict(a=Grid(None, 'x'), b=Grid(2, 3))))
    assert 4 == len(dicts)
    dicts = list(Space(TuningParametersTemplate(dict(a=1, b=[Grid(2, 3), Grid(4, 5)]))))
    assert 4 == len(dicts)
    assert dict(a=1, b=[2, 4]) == dicts[0]
    assert dict(a=1, b=[2, 5]) == dicts[1]
    assert dict(a=1, b=[3, 4]) == dicts[2]
    assert dict(a=1, b=[3, 5]) == dicts[3]
    dicts = list(Space(a=1, b=dict(x=Grid(2, 3), y=Grid(4, 5))))
    assert 4 == len(dicts)
    assert dict(a=1, b=dict(x=2, y=4)) == dicts[0]
    assert dict(a=1, b=dict(x=2, y=5)) == dicts[1]
    assert dict(a=1, b=dict(x=3, y=4)) == dicts[2]
    assert dict(a=1, b=dict(x=3, y=5)) == dicts[3]