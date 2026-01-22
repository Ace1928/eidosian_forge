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
def _operator_dunder_methods(self, dunder: str) -> Set[Callable]:
    if dunder not in self._operation_methods_cache:
        self._operation_methods_cache[dunder] = self._safe_get_methods(self.allowed_operations, dunder)
    return self._operation_methods_cache[dunder]