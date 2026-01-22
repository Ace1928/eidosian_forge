import sys
from configparser import ConfigParser
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type as TypingType, Union
from mypy.errorcodes import ErrorCode
from mypy.nodes import (
from mypy.options import Options
from mypy.plugin import (
from mypy.plugins import dataclasses
from mypy.semanal import set_callable_name  # type: ignore
from mypy.server.trigger import make_wildcard_trigger
from mypy.types import (
from mypy.typevars import fill_typevars
from mypy.util import get_unique_redefinition_name
from mypy.version import __version__ as mypy_version
from pydantic.utils import is_valid_field
def add_construct_method(self, fields: List['PydanticModelField']) -> None:
    """
        Adds a fully typed `construct` classmethod to the class.

        Similar to the fields-aware __init__ method, but always uses the field names (not aliases),
        and does not treat settings fields as optional.
        """
    ctx = self._ctx
    set_str = ctx.api.named_type(f'{BUILTINS_NAME}.set', [ctx.api.named_type(f'{BUILTINS_NAME}.str')])
    optional_set_str = UnionType([set_str, NoneType()])
    fields_set_argument = Argument(Var('_fields_set', optional_set_str), optional_set_str, None, ARG_OPT)
    construct_arguments = self.get_field_arguments(fields, typed=True, force_all_optional=False, use_alias=False)
    construct_arguments = [fields_set_argument] + construct_arguments
    obj_type = ctx.api.named_type(f'{BUILTINS_NAME}.object')
    self_tvar_name = '_PydanticBaseModel'
    tvar_fullname = ctx.cls.fullname + '.' + self_tvar_name
    if MYPY_VERSION_TUPLE >= (1, 4):
        tvd = TypeVarType(self_tvar_name, tvar_fullname, -1, [], obj_type, AnyType(TypeOfAny.from_omitted_generics))
        self_tvar_expr = TypeVarExpr(self_tvar_name, tvar_fullname, [], obj_type, AnyType(TypeOfAny.from_omitted_generics))
    else:
        tvd = TypeVarDef(self_tvar_name, tvar_fullname, -1, [], obj_type)
        self_tvar_expr = TypeVarExpr(self_tvar_name, tvar_fullname, [], obj_type)
    ctx.cls.info.names[self_tvar_name] = SymbolTableNode(MDEF, self_tvar_expr)
    if isinstance(tvd, TypeVarType):
        self_type = tvd
    else:
        self_type = TypeVarType(tvd)
    add_method(ctx, 'construct', construct_arguments, return_type=self_type, self_type=self_type, tvar_def=tvd, is_classmethod=True)