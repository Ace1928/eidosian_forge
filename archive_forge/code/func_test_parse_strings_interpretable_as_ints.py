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
def test_parse_strings_interpretable_as_ints():
    """Test whether strings interpretable as integers are parsed correctly (i. e. as strings)."""
    cfg = Config().from_str(f'[a]\nfoo = [${{b.bar}}, "00${{b.bar}}", "y"]\n\n[b]\nbar = 3')
    assert cfg['a']['foo'] == [3, '003', 'y']
    assert cfg['b']['bar'] == 3