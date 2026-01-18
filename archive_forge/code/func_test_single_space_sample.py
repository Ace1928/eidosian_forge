from tune import Choice, Grid, Rand, RandInt, Space, TuningParametersTemplate, FuncParam
from pytest import raises
def test_single_space_sample():
    assert not Space(a=1).has_stochastic
    assert not Space(a=1, b=Grid(1, 2)).has_stochastic
    assert Space(a=1, b=[Grid(1, 2), Rand(0.0, 1.0)]).has_stochastic
    dicts = list(Space(a=1, b=Grid(1, 2)).sample(100))
    assert 2 == len(dicts)
    dicts = list(Space(a=1, b=RandInt(1, 2)).sample(100))
    assert 100 == len(dicts)
    space = Space(a=1, b=[Grid(1, 2), Rand(0.0, 1.0)], c=Choice('a', 'b'))
    assert list(space.sample(5, 0)) == list(space.sample(5, 0))
    assert list(space.sample(5, 0)) != list(space.sample(5, 1))
    dicts = list(space.sample(5, 0))
    assert 10 == len(dicts)
    assert 5 == len(set((d.template['b'][1] for d in dicts)))