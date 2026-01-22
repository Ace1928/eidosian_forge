import builtins
import collections
import dis
import operator
import logging
import textwrap
from numba.core import errors, ir, config
from numba.core.errors import NotDefinedError, UnsupportedError, error_extras
from numba.core.ir_utils import get_definition, guard
from numba.core.utils import (PYVERSION, BINOPS_TO_OPERATORS,
from numba.core.byteflow import Flow, AdaptDFA, AdaptCFA, BlockKind
from numba.core.unsafe import eh
from numba.cpython.unsafe.tuple import unpack_single_tuple
def _call_function_ex_replace_args_small(old_body, tuple_expr, new_body, buildtuple_idx, func_ir, already_deleted_defs):
    """
    Extracts the args passed as vararg
    for CALL_FUNCTION_EX. This pass is taken when
    n_args <= 30 and the bytecode looks like:

        # Start for each argument
        LOAD_FAST  # Load each argument.
        # End for each argument
        ...
        BUILD_TUPLE # Create a tuple of the arguments

    In the IR generated, the vararg refer
    to a single build_tuple that contains all of the
    args. In addition to returning the args, this
    function updates new_body to remove all usage
    of the tuple.
    """
    new_body[buildtuple_idx] = None
    _remove_assignment_definition(old_body, buildtuple_idx, func_ir, already_deleted_defs)
    return tuple_expr.items