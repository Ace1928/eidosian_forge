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
def test_registry_methods():
    with pytest.raises(ValueError):
        my_registry.get('dfkoofkds', 'catsie.v1')
    my_registry.cats.register('catsie.v123')(None)
    with pytest.raises(ValueError):
        my_registry.get('cats', 'catsie.v123')