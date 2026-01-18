import numpy
import pytest
from thinc.api import Optimizer, registry
@pytest.mark.parametrize('name', ['RAdam.v1', 'Adam.v1', 'SGD.v1'])
def test_optimizers_from_config(name):
    learn_rate = 0.123
    cfg = {'@optimizers': name, 'learn_rate': learn_rate}
    optimizer = registry.resolve({'config': cfg})['config']
    assert optimizer.learn_rate == learn_rate