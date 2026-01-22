from inspect import isclass, signature, Signature
from typing import (
import ast
import builtins
import collections
import operator
import sys
from functools import cached_property
from dataclasses import dataclass, field
from types import MethodDescriptorType, ModuleType
from IPython.utils.decorators import undoc
class _Duck:
    """A dummy class used to create objects pretending to have given attributes"""

    def __init__(self, attributes: Optional[dict]=None, items: Optional[dict]=None):
        self.attributes = attributes or {}
        self.items = items or {}

    def __getattr__(self, attr: str):
        return self.attributes[attr]

    def __hasattr__(self, attr: str):
        return attr in self.attributes

    def __dir__(self):
        return [*dir(super), *self.attributes]

    def __getitem__(self, key: str):
        return self.items[key]

    def __hasitem__(self, key: str):
        return self.items[key]

    def _ipython_key_completions_(self):
        return self.items.keys()