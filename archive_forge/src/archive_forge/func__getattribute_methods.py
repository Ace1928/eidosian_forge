from inspect import signature, Signature
from typing import (
import ast
import builtins
import collections
import operator
import sys
from functools import cached_property
from dataclasses import dataclass, field
from types import MethodDescriptorType, ModuleType
from IPython.utils.docs import GENERATING_DOCUMENTATION
from IPython.utils.decorators import undoc
@cached_property
def _getattribute_methods(self) -> Set[Callable]:
    return self._safe_get_methods(self.allowed_getattr, '__getattribute__')