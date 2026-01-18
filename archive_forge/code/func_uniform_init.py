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
def uniform_init(shape: Tuple[int, ...], *, lo: float=-0.1, hi: float=0.1) -> List[float]:
    return numpy.random.uniform(lo, hi, shape).tolist()