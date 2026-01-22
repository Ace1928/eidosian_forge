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
def _call_function_ex_replace_kws_small(old_body, keyword_expr, new_body, buildmap_idx, func_ir, already_deleted_defs):
    """
    Extracts the kws args passed as varkwarg
    for CALL_FUNCTION_EX. This pass is taken when
    n_kws <= 15 and the bytecode looks like:

        # Start for each argument
        LOAD_FAST  # Load each argument.
        # End for each argument
        ...
        BUILD_CONST_KEY_MAP # Build a map

    In the generated IR, the varkwarg refers
    to a single build_map that contains all of the
    kws. In addition to returning the kws, this
    function updates new_body to remove all usage
    of the map.
    """
    kws = keyword_expr.items.copy()
    value_indexes = keyword_expr.value_indexes
    for key, index in value_indexes.items():
        kws[index] = (key, kws[index][1])
    new_body[buildmap_idx] = None
    _remove_assignment_definition(old_body, buildmap_idx, func_ir, already_deleted_defs)
    return kws