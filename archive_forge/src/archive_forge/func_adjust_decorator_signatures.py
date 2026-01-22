from __future__ import annotations
import sys
from configparser import ConfigParser
from typing import Any, Callable, Iterator
from mypy.errorcodes import ErrorCode
from mypy.expandtype import expand_type, expand_type_by_instance
from mypy.nodes import (
from mypy.options import Options
from mypy.plugin import (
from mypy.plugins import dataclasses
from mypy.plugins.common import (
from mypy.semanal import set_callable_name
from mypy.server.trigger import make_wildcard_trigger
from mypy.state import state
from mypy.typeops import map_type_from_supertype
from mypy.types import (
from mypy.typevars import fill_typevars
from mypy.util import get_unique_redefinition_name
from mypy.version import __version__ as mypy_version
from pydantic._internal import _fields
from pydantic.version import parse_mypy_version
def adjust_decorator_signatures(self) -> None:
    """When we decorate a function `f` with `pydantic.validator(...)`, `pydantic.field_validator`
        or `pydantic.serializer(...)`, mypy sees `f` as a regular method taking a `self` instance,
        even though pydantic internally wraps `f` with `classmethod` if necessary.

        Teach mypy this by marking any function whose outermost decorator is a `validator()`,
        `field_validator()` or `serializer()` call as a `classmethod`.
        """
    for name, sym in self._cls.info.names.items():
        if isinstance(sym.node, Decorator):
            first_dec = sym.node.original_decorators[0]
            if isinstance(first_dec, CallExpr) and isinstance(first_dec.callee, NameExpr) and (first_dec.callee.fullname in DECORATOR_FULLNAMES) and (not (first_dec.callee.fullname == MODEL_VALIDATOR_FULLNAME and any((first_dec.arg_names[i] == 'mode' and isinstance(arg, StrExpr) and (arg.value == 'after') for i, arg in enumerate(first_dec.args))))):
                sym.node.func.is_class = True