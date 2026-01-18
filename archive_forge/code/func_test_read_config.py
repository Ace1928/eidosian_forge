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
def test_read_config():
    byte_string = EXAMPLE_CONFIG.encode('utf8')
    cfg = Config().from_bytes(byte_string)
    assert cfg['optimizer']['beta1'] == 0.9
    assert cfg['optimizer']['learn_rate']['initial_rate'] == 0.1
    assert cfg['pipeline']['classifier']['factory'] == 'classifier'
    assert cfg['pipeline']['classifier']['model']['embedding']['width'] == 128