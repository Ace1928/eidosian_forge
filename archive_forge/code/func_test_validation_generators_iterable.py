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
def test_validation_generators_iterable():

    @my_registry.optimizers('test_optimizer.v1')
    def test_optimizer_v1(rate: float) -> None:
        return None

    @my_registry.schedules('test_schedule.v1')
    def test_schedule_v1(some_value: float=1.0) -> Iterable[float]:
        while True:
            yield some_value
    config = {'optimizer': {'@optimizers': 'test_optimizer.v1', 'rate': 0.1}}
    my_registry.resolve(config)