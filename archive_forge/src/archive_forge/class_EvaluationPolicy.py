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
@undoc
@dataclass
class EvaluationPolicy:
    """Definition of evaluation policy."""
    allow_locals_access: bool = False
    allow_globals_access: bool = False
    allow_item_access: bool = False
    allow_attr_access: bool = False
    allow_builtins_access: bool = False
    allow_all_operations: bool = False
    allow_any_calls: bool = False
    allowed_calls: Set[Callable] = field(default_factory=set)

    def can_get_item(self, value, item):
        return self.allow_item_access

    def can_get_attr(self, value, attr):
        return self.allow_attr_access

    def can_operate(self, dunders: Tuple[str, ...], a, b=None):
        if self.allow_all_operations:
            return True

    def can_call(self, func):
        if self.allow_any_calls:
            return True
        if func in self.allowed_calls:
            return True
        owner_method = _unbind_method(func)
        if owner_method and owner_method in self.allowed_calls:
            return True