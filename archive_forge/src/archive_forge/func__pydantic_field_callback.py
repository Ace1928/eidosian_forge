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
def _pydantic_field_callback(self, ctx: FunctionContext) -> 'Type':
    """
        Extract the type of the `default` argument from the Field function, and use it as the return type.

        In particular:
        * Check whether the default and default_factory argument is specified.
        * Output an error if both are specified.
        * Retrieve the type of the argument which is specified, and use it as return type for the function.
        """
    default_any_type = ctx.default_return_type
    assert ctx.callee_arg_names[0] == 'default', '"default" is no longer first argument in Field()'
    assert ctx.callee_arg_names[1] == 'default_factory', '"default_factory" is no longer second argument in Field()'
    default_args = ctx.args[0]
    default_factory_args = ctx.args[1]
    if default_args and default_factory_args:
        error_default_and_default_factory_specified(ctx.api, ctx.context)
        return default_any_type
    if default_args:
        default_type = ctx.arg_types[0][0]
        default_arg = default_args[0]
        if not isinstance(default_arg, EllipsisExpr):
            return default_type
    elif default_factory_args:
        default_factory_type = ctx.arg_types[1][0]
        if isinstance(default_factory_type, Overloaded):
            if MYPY_VERSION_TUPLE > (0, 910):
                default_factory_type = default_factory_type.items[0]
            else:
                default_factory_type = default_factory_type.items()[0]
        if isinstance(default_factory_type, CallableType):
            ret_type = default_factory_type.ret_type
            args = getattr(ret_type, 'args', None)
            if args:
                if all((isinstance(arg, TypeVarType) for arg in args)):
                    ret_type.args = tuple((default_any_type for _ in args))
            return ret_type
    return default_any_type