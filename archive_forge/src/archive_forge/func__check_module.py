import functools
import importlib
import os
import warnings
from functools import lru_cache
from importlib.util import find_spec
from types import ModuleType
from typing import Any, Callable, List, Optional, TypeVar
import pkg_resources
from packaging.requirements import Requirement
from packaging.version import Version
from typing_extensions import ParamSpec
def _check_module(self) -> None:
    assert self.module
    self.available = module_available(self.module)
    if self.available:
        self.message = f'Module {self.module!r} available'
    else:
        self.message = f'Module not found: {self.module!r}. HINT: Try running `pip install -U {self.module}`'