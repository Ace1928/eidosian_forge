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
def _find_dunder(node_op, dunders) -> Union[Tuple[str, ...], None]:
    dunder = None
    for op, candidate_dunder in dunders.items():
        if isinstance(node_op, op):
            dunder = candidate_dunder
    return dunder