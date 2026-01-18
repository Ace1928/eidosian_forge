import inspect
import pickle
import platform
from types import GeneratorType
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import catalogue
import pytest
from confection import Config, ConfigValidationError
from confection.tests.util import Cat, make_tempdir, my_registry
from confection.util import Generator, partial
def test_partials_from_config():
    """Test that functions registered with partial applications are handled
    correctly (e.g. initializers)."""
    numpy = pytest.importorskip('numpy')

    def uniform_init(shape: Tuple[int, ...], *, lo: float=-0.1, hi: float=0.1) -> List[float]:
        return numpy.random.uniform(lo, hi, shape).tolist()

    @my_registry.initializers('uniform_init.v1')
    def configure_uniform_init(*, lo: float=-0.1, hi: float=0.1) -> Callable[[List[float]], List[float]]:
        return partial(uniform_init, lo=lo, hi=hi)
    name = 'uniform_init.v1'
    cfg = {'test': {'@initializers': name, 'lo': -0.2}}
    func = my_registry.resolve(cfg)['test']
    assert hasattr(func, '__call__')
    assert len(inspect.signature(func).parameters) == 3
    assert inspect.signature(func).parameters['lo'].default == -0.2
    assert numpy.asarray(func((2, 3))).shape == (2, 3)
    bad_cfg = {'test': {'@initializers': name, 'lo': [0.5]}}
    with pytest.raises(ConfigValidationError):
        my_registry.resolve(bad_cfg)
    bad_cfg = {'test': {'@initializers': name, 'lo': -0.2, 'other': 10}}
    with pytest.raises(ConfigValidationError):
        my_registry.resolve(bad_cfg)