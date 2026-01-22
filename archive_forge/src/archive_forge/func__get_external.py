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
def _get_external(module_name: str, access_path: Sequence[str]):
    """Get value from external module given a dotted access path.

    Raises:
    * `KeyError` if module is removed not found, and
    * `AttributeError` if acess path does not match an exported object
    """
    member_type = sys.modules[module_name]
    for attr in access_path:
        member_type = getattr(member_type, attr)
    return member_type