import collections
import pickle
import pytest
import networkx as nx
from networkx.utils.configs import Config
def test_not_strict():

    class FlexibleConfig(Config, strict=False):
        x: int
    cfg = FlexibleConfig(x=1)
    assert '_strict' not in cfg
    assert len(cfg) == 1
    assert list(cfg) == ['x']
    assert list(cfg.keys()) == ['x']
    assert list(cfg.values()) == [1]
    assert list(cfg.items()) == [('x', 1)]
    assert cfg.x == 1
    assert cfg['x'] == 1
    assert 'x' in cfg
    assert hasattr(cfg, 'x')
    assert 'FlexibleConfig(x=1)' in repr(cfg)
    assert cfg == FlexibleConfig(x=1)
    del cfg.x
    assert 'FlexibleConfig()' in repr(cfg)
    assert len(cfg) == 0
    assert not hasattr(cfg, 'x')
    assert 'x' not in cfg
    assert not hasattr(cfg, 'y')
    assert 'y' not in cfg
    cfg.y = 2
    assert len(cfg) == 1
    assert list(cfg) == ['y']
    assert list(cfg.keys()) == ['y']
    assert list(cfg.values()) == [2]
    assert list(cfg.items()) == [('y', 2)]
    assert cfg.y == 2
    assert cfg['y'] == 2
    assert hasattr(cfg, 'y')
    assert 'y' in cfg
    del cfg['y']
    assert len(cfg) == 0
    assert list(cfg) == []
    with pytest.raises(AttributeError, match='y'):
        del cfg.y
    with pytest.raises(KeyError, match='y'):
        del cfg['y']
    with pytest.raises(TypeError, match='missing 1 required keyword-only'):
        FlexibleConfig()
    with pytest.raises(TypeError, match="unexpected keyword argument 'y'"):
        FlexibleConfig(x=1, y=2)

    class FlexibleConfigWithDefault(Config, strict=False):
        x: int = 0
    assert FlexibleConfigWithDefault().x == 0
    assert FlexibleConfigWithDefault(x=1)['x'] == 1