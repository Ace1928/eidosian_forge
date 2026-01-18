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
def test_resolve_no_schema():
    config = {'one': 1, 'two': {'three': {'@cats': 'catsie.v1', 'evil': True}}}
    result = my_registry.resolve({'cfg': config})['cfg']
    assert result['one'] == 1
    assert result['two'] == {'three': 'scratch!'}
    with pytest.raises(ConfigValidationError):
        config = {'two': {'three': {'@cats': 'catsie.v1', 'evil': 'true'}}}
        my_registry.resolve(config)