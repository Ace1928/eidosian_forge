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
@my_registry.schedules('test_schedule.v1')
def test_schedule_v1(some_value: float=1.0) -> Iterable[float]:
    while True:
        yield some_value