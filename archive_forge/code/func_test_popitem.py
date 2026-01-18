import pytest
import rpy2.robjects as robjects
import array
def test_popitem():
    env = robjects.Environment()
    env['a'] = 123
    env['b'] = 456
    robjs = []
    assert len(env) == 2
    robjs.append(env.popitem())
    assert len(env) == 1
    robjs.append(env.popitem())
    assert len(env) == 0
    assert sorted([(k, v[0]) for k, v in robjs]) == [('a', 123), ('b', 456)]
    with pytest.raises(KeyError):
        robjs.append(env.popitem())