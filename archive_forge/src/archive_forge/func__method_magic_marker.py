import os
import re
import sys
from getopt import getopt, GetoptError
from traitlets.config.configurable import Configurable
from . import oinspect
from .error import UsageError
from .inputtransformer2 import ESC_MAGIC, ESC_MAGIC2
from ..utils.ipstruct import Struct
from ..utils.process import arg_split
from ..utils.text import dedent
from traitlets import Bool, Dict, Instance, observe
from logging import error
import typing as t
def _method_magic_marker(magic_kind):
    """Decorator factory for methods in Magics subclasses.
    """
    validate_type(magic_kind)

    def magic_deco(arg):
        if callable(arg):
            func = arg
            name = func.__name__
            retval = arg
            record_magic(magics, magic_kind, name, name)
        elif isinstance(arg, str):
            name = arg

            def mark(func, *a, **kw):
                record_magic(magics, magic_kind, name, func.__name__)
                return func
            retval = mark
        else:
            raise TypeError('Decorator can only be called with string or function')
        return retval
    magic_deco.__doc__ = _docstring_template.format('method', magic_kind)
    return magic_deco