import inspect
import pickle
from types import GeneratorType
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
import catalogue
import numpy
import pytest
import thinc.config
from thinc.api import Config, Model, NumpyOps, RAdam
from thinc.config import ConfigValidationError
from thinc.types import Generator, Ragged
from thinc.util import partial
from .util import make_tempdir
def test_objects_from_config():
    config = {'optimizer': {'@optimizers': 'my_cool_optimizer.v1', 'beta1': 0.2, 'learn_rate': {'@schedules': 'my_cool_repetitive_schedule.v1', 'base_rate': 0.001, 'repeat': 4}}}

    @thinc.registry.optimizers.register('my_cool_optimizer.v1')
    def make_my_optimizer(learn_rate: List[float], beta1: float):
        return RAdam(learn_rate, beta1=beta1)

    @thinc.registry.schedules('my_cool_repetitive_schedule.v1')
    def decaying(base_rate: float, repeat: int) -> List[float]:
        return repeat * [base_rate]
    optimizer = my_registry.resolve(config)['optimizer']
    assert optimizer.b1 == 0.2
    assert 'learn_rate' in optimizer.schedules
    assert optimizer.learn_rate == 0.001