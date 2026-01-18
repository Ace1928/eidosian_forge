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
def test_resolve_schema_coerced():

    class TestBaseSchema(BaseModel):
        test1: str
        test2: bool
        test3: float

    class TestSchema(BaseModel):
        cfg: TestBaseSchema
    config = {'test1': 123, 'test2': 1, 'test3': 5}
    filled = my_registry.fill({'cfg': config}, schema=TestSchema)
    result = my_registry.resolve({'cfg': config}, schema=TestSchema)
    assert result['cfg'] == {'test1': '123', 'test2': True, 'test3': 5.0}
    assert filled['cfg'] == config