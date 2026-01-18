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
def test_validation_bad_function():

    @my_registry.optimizers('bad.v1')
    def bad() -> None:
        raise ValueError('This is an error in the function')
        return None

    @my_registry.optimizers('good.v1')
    def good() -> None:
        return None
    config = {'test': {'@optimizers': 'bad.v1'}}
    with pytest.raises(ValueError):
        my_registry.resolve(config)
    config = {'test': {'@optimizers': 'good.v1', 'invalid_arg': 1}}
    with pytest.raises(ConfigValidationError):
        my_registry.resolve(config)