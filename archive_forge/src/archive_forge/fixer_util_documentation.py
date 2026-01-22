from lib2to3.fixer_util import (FromImport, Newline, is_import,
from lib2to3.pytree import Leaf, Node
from lib2to3.pygram import python_symbols as syms
from lib2to3.pygram import token
import re

    Example:
    >>> wrap_in_fn_call("oldstr", (arg,))
    oldstr(arg)

    >>> wrap_in_fn_call("olddiv", (arg1, arg2))
    olddiv(arg1, arg2)

    >>> wrap_in_fn_call("olddiv", [arg1, comma, arg2, comma, arg3])
    olddiv(arg1, arg2, arg3)
    