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
def test_resolve_prefilled_values():

    class Language(object):

        def __init__(self):
            ...

    @my_registry.optimizers('prefilled.v1')
    def prefilled(nlp: Language, value: int=10):
        return (nlp, value)
    config = {'test': {'@optimizers': 'prefilled.v1', 'nlp': Language(), 'value': 50}}
    resolved = my_registry.resolve(config, validate=True)
    result = resolved['test']
    assert isinstance(result[0], Language)
    assert result[1] == 50