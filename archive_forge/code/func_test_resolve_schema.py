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
def test_resolve_schema():

    class TestBaseSubSchema(BaseModel):
        three: str

    class TestBaseSchema(BaseModel):
        one: PositiveInt
        two: TestBaseSubSchema

        class Config:
            extra = 'forbid'

    class TestSchema(BaseModel):
        cfg: TestBaseSchema
    config = {'one': 1, 'two': {'three': {'@cats': 'catsie.v1', 'evil': True}}}
    my_registry.resolve({'cfg': config}, schema=TestSchema)
    config = {'one': -1, 'two': {'three': {'@cats': 'catsie.v1', 'evil': True}}}
    with pytest.raises(ConfigValidationError):
        my_registry.resolve({'cfg': config}, schema=TestSchema)
    config = {'one': 1, 'two': {'four': {'@cats': 'catsie.v1', 'evil': True}}}
    with pytest.raises(ConfigValidationError):
        my_registry.resolve({'cfg': config}, schema=TestSchema)