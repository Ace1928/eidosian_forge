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
def test_resolve_overrides():
    config = {'cfg': {'one': 1, 'two': {'three': {'@cats': 'catsie.v1', 'evil': True, 'cute': False}}}}
    overrides = {'cfg.two.three.evil': False}
    result = my_registry.resolve(config, overrides=overrides, validate=True)
    assert result['cfg']['two']['three'] == 'meow'
    overrides = {'cfg.two.three': 3}
    result = my_registry.resolve(config, overrides=overrides, validate=True)
    assert result['cfg']['two']['three'] == 3
    overrides = {'cfg': {'one': {'@cats': 'catsie.v1', 'evil': False}, 'two': None}}
    result = my_registry.resolve(config, overrides=overrides)
    assert result['cfg']['one'] == 'meow'
    assert result['cfg']['two'] is None
    with pytest.raises(ConfigValidationError):
        overrides = {'cfg.two.three.evil': 20}
        my_registry.resolve(config, overrides=overrides, validate=True)
    with pytest.raises(ConfigValidationError):
        overrides = {'cfg': {'one': {'@cats': 'catsie.v1'}, 'two': None}}
        my_registry.resolve(config, overrides=overrides)
    with pytest.raises(ConfigValidationError):
        overrides = {'cfg.two.three.evil': False, 'cfg.two.four': True}
        my_registry.resolve(config, overrides=overrides, validate=True)
    with pytest.raises(ConfigValidationError):
        overrides = {'cfg.five': False}
        my_registry.resolve(config, overrides=overrides, validate=True)