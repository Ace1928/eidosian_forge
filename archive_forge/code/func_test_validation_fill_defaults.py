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
def test_validation_fill_defaults():
    config = {'cfg': {'one': 1, 'two': {'@cats': 'catsie.v1', 'evil': 'hello'}}}
    result = my_registry.fill(config, validate=False)
    assert len(result['cfg']['two']) == 3
    with pytest.raises(ConfigValidationError):
        my_registry.fill(config)
    config = {'cfg': {'one': 1, 'two': {'@cats': 'catsie.v2', 'evil': False}}}
    result = my_registry.fill(config)
    assert len(result['cfg']['two']) == 4
    assert result['cfg']['two']['evil'] is False
    assert result['cfg']['two']['cute'] is True
    assert result['cfg']['two']['cute_level'] == 1