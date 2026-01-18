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
def test_validate_promise():
    config = {'required': 1, 'optional': good_catsie}
    filled, _, validated = my_registry._fill(config, DefaultsSchema)
    assert filled == config
    assert validated == {'required': 1, 'optional': 'meow'}