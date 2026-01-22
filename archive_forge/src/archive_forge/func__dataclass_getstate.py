import re
import sys
import copy
import types
import inspect
import keyword
import builtins
import functools
import itertools
import abc
import _thread
from types import FunctionType, GenericAlias
def _dataclass_getstate(self):
    return [getattr(self, f.name) for f in fields(self)]