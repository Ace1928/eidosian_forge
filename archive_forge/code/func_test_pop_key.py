import pytest
import rpy2.robjects as robjects
import array
def test_pop_key():
    env = robjects.Environment()
    env['a'] = 123
    env['b'] = 456
    robjs = []
    assert len(env) == 2
    robjs.append(env.pop('a'))
    assert len(env) == 1
    robjs.append(env.pop('b'))
    assert len(env) == 0
    assert [x[0] for x in robjs] == [123, 456]
    with pytest.raises(KeyError):
        env.pop('c')
    assert env.pop('c', 789) == 789
    with pytest.raises(ValueError):
        env.pop('c', 1, 2)