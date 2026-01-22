from pyarrow._compute import (  # noqa
from collections import namedtuple
import inspect
from textwrap import dedent
import warnings
import pyarrow as pa
from pyarrow import _compute_docstrings
from pyarrow.vendored import docscrape
def _make_global_functions():
    """
    Make global functions wrapping each compute function.

    Note that some of the automatically-generated wrappers may be overridden
    by custom versions below.
    """
    g = globals()
    reg = function_registry()
    rewrites = {'and': 'and_', 'or': 'or_'}
    for cpp_name in reg.list_functions():
        name = rewrites.get(cpp_name, cpp_name)
        func = reg.get_function(cpp_name)
        if func.kind == 'hash_aggregate':
            continue
        if func.kind == 'scalar_aggregate' and func.arity == 0:
            continue
        assert name not in g, name
        g[cpp_name] = g[name] = _wrap_function(name, func)