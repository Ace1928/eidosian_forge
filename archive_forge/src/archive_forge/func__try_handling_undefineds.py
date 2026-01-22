import functools
import sys
import traceback
import numpy as np
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.autograph.operators import variables
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.autograph.utils import misc
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.autograph.utils import type_registry
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.types import distribute
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
def _try_handling_undefineds(body, get_state, set_state, init_vars, nulls, shape_invariants, symbol_names):
    """Makes a best-effort attempt to substitute undefineds with placeholders.

  Note: this substitution requires two things to happen:
   1. the types of loop variables could be inferred (usually by staging one
       iteration)
   2. these types could be replaced by placeholders (e.g. zero values, for
       tensors).

  Args:
    body: a function representing the loop body. See while_stmt.
    get_state: state getter for the loop statement. See while_stmt.
    set_state: state getter for the loop statement. See while_stmt.
    init_vars: loop variables before entering the loop. See while_stmt.
    nulls: list of boolean flags indicating whether the corresponding loop var
      is None or undefined.
    shape_invariants: user-specified shape invariant for each loop variable.
    symbol_names: list of loop variable names. See while_stmt.

  Returns:
    A tuple (success, new_init_vars, extra_shape_invariants, failure_message):
     * success is a boolean flag indicating
       whether types could be successfully inferred (step 1 above)
     * new_init_vars contains the loop vars, with None or undefined values
       replaced by default values, where possible (step 2 above)
     * extra_shape_invariants contains shape invariants that would be needed
       by while_stmt, for instance if the placeholder values had a shape
       different from the corresponding loop outputs
  """
    state_modified = False
    first_iter_vars = None
    failure_message = None
    try:
        with func_graph.FuncGraph('tmp').as_default():

            def autocast_to_tensor(v):
                if isinstance(v, (int, float, bool, str, list, tuple, np.ndarray, np.generic)):
                    init_val = tensor_conversion.convert_to_tensor_v2(v)
                    return array_ops.placeholder(init_val.dtype, init_val.shape)
                return v
            autocast_init_vars = nest.map_structure(autocast_to_tensor, init_vars)
            set_state(autocast_init_vars)
            state_modified = True
            body()
            first_iter_vars = get_state()
        inits_and_invariants = tuple((_placeholder_value(iv, i, v) if n else (v, None) for v, n, iv, i in zip(init_vars, nulls, first_iter_vars, shape_invariants)))
        init_vars, extra_shape_invariants = zip(*inits_and_invariants)
        success = True
    except (UnboundLocalError, TypeError, ValueError, KeyError):
        ag_logging.log(1, 'Caught error while staging loop body', exc_info=True)
        exc = sys.exc_info()
        failure_message = 'Note: AutoGraph tried to define it automatically, but ran into a {}: {}'.format(exc[0].__name__, exc[1])
    finally:
        if state_modified:
            set_state(init_vars)
    verify_loop_init_vars(init_vars, symbol_names, first_iter_vars, extra_message=failure_message)
    return (success, init_vars, extra_shape_invariants)