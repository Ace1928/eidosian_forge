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
class EvaluationContext(NamedTuple):
    locals: dict
    globals: dict
    evaluation: Literal['forbidden', 'minimal', 'limited', 'unsafe', 'dangerous'] = 'forbidden'
    in_subscript: bool = False