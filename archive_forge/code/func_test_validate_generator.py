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
def test_validate_generator():
    """Test that generator replacement for validation in config doesn't
    actually replace the returned value."""

    @my_registry.schedules('test_schedule.v2')
    def test_schedule():
        while True:
            yield 10
    cfg = {'@schedules': 'test_schedule.v2'}
    result = my_registry.resolve({'test': cfg})['test']
    assert isinstance(result, GeneratorType)

    @my_registry.optimizers('test_optimizer.v2')
    def test_optimizer2(rate: Generator) -> Generator:
        return rate
    cfg = {'@optimizers': 'test_optimizer.v2', 'rate': {'@schedules': 'test_schedule.v2'}}
    result = my_registry.resolve({'test': cfg})['test']
    assert isinstance(result, GeneratorType)

    @my_registry.optimizers('test_optimizer.v3')
    def test_optimizer3(schedules: Dict[str, Generator]) -> Generator:
        return schedules['rate']
    cfg = {'@optimizers': 'test_optimizer.v3', 'schedules': {'rate': {'@schedules': 'test_schedule.v2'}}}
    result = my_registry.resolve({'test': cfg})['test']
    assert isinstance(result, GeneratorType)

    @my_registry.optimizers('test_optimizer.v4')
    def test_optimizer4(*schedules: Generator) -> Generator:
        return schedules[0]