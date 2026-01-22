from pyarrow._compute import (  # noqa
from collections import namedtuple
import inspect
from textwrap import dedent
import warnings
import pyarrow as pa
from pyarrow import _compute_docstrings
from pyarrow.vendored import docscrape
def _make_generic_wrapper(func_name, func, options_class, arity):
    if options_class is None:

        def wrapper(*args, memory_pool=None):
            if arity is not Ellipsis and len(args) != arity:
                raise TypeError(f'{func_name} takes {arity} positional argument(s), but {len(args)} were given')
            if args and isinstance(args[0], Expression):
                return Expression._call(func_name, list(args))
            return func.call(args, None, memory_pool)
    else:

        def wrapper(*args, memory_pool=None, options=None, **kwargs):
            if arity is not Ellipsis:
                if len(args) < arity:
                    raise TypeError(f'{func_name} takes {arity} positional argument(s), but {len(args)} were given')
                option_args = args[arity:]
                args = args[:arity]
            else:
                option_args = ()
            options = _handle_options(func_name, options_class, options, option_args, kwargs)
            if args and isinstance(args[0], Expression):
                return Expression._call(func_name, list(args), options)
            return func.call(args, options, memory_pool)
    return wrapper