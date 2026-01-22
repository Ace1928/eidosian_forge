import os
import re
import typing as t
from gettext import gettext as _
from .core import Argument
from .core import BaseCommand
from .core import Context
from .core import MultiCommand
from .core import Option
from .core import Parameter
from .core import ParameterSource
from .parser import split_arg_string
from .utils import echo
def _is_incomplete_option(ctx: Context, args: t.List[str], param: Parameter) -> bool:
    """Determine if the given parameter is an option that needs a value.

    :param args: List of complete args before the incomplete value.
    :param param: Option object being checked.
    """
    if not isinstance(param, Option):
        return False
    if param.is_flag or param.count:
        return False
    last_option = None
    for index, arg in enumerate(reversed(args)):
        if index + 1 > param.nargs:
            break
        if _start_of_option(ctx, arg):
            last_option = arg
    return last_option is not None and last_option in param.opts