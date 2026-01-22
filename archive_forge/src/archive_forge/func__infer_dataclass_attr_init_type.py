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
def _infer_dataclass_attr_init_type(self, sym: SymbolTableNode, name: str, context: Context) -> Type | None:
    """Infer __init__ argument type for an attribute.

        In particular, possibly use the signature of __set__.
        """
    default = sym.type
    if sym.implicit:
        return default
    t = get_proper_type(sym.type)
    if not isinstance(t, Instance):
        return default
    setter = t.type.get('__set__')
    if setter:
        if isinstance(setter.node, FuncDef):
            super_info = t.type.get_containing_type_info('__set__')
            assert super_info
            if setter.type:
                setter_type = get_proper_type(map_type_from_supertype(setter.type, t.type, super_info))
            else:
                return AnyType(TypeOfAny.unannotated)
            if isinstance(setter_type, CallableType) and setter_type.arg_kinds == [ARG_POS, ARG_POS, ARG_POS]:
                return expand_type_by_instance(setter_type.arg_types[2], t)
            else:
                self._api.fail(f'Unsupported signature for "__set__" in "{t.type.name}"', context)
        else:
            self._api.fail(f'Unsupported "__set__" in "{t.type.name}"', context)
    return default