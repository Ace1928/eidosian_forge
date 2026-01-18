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
def test_partials_from_config_nested():
    """Test that partial functions are passed correctly to other registered
    functions that consume them (e.g. initializers -> layers)."""

    def test_initializer(a: int, b: int=1) -> int:
        return a * b

    @my_registry.initializers('test_initializer.v1')
    def configure_test_initializer(b: int=1) -> Callable[[int], int]:
        return partial(test_initializer, b=b)

    @my_registry.layers('test_layer.v1')
    def test_layer(init: Callable[[int], int], c: int=1) -> Callable[[int], int]:
        return lambda x: x + init(c)
    cfg = {'@layers': 'test_layer.v1', 'c': 5, 'init': {'@initializers': 'test_initializer.v1', 'b': 10}}
    func = my_registry.resolve({'test': cfg})['test']
    assert func(1) == 51
    assert func(100) == 150