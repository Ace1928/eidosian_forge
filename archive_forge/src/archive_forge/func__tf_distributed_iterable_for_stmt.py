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
def _tf_distributed_iterable_for_stmt(iter_, extra_test, body, get_state, set_state, symbol_names, opts):
    """Overload of for_stmt that iterates over TF distributed datasets."""
    if extra_test is not None:
        raise NotImplementedError('break and return statements are not yet supported in for ... in distributed input loops.')
    init_vars = get_state()
    verify_loop_init_vars(init_vars, symbol_names)
    if 'shape_invariants' in opts:
        opts['shape_invariants'] = _shape_invariants_mapping_to_positional_list(opts['shape_invariants'], init_vars)

    def reduce_body(loop_vars, iterate):
        set_state(loop_vars)
        body(iterate)
        new_loop_vars = get_state()
        verify_tf_loop_vars(init_vars, loop_vars, new_loop_vars, symbol_names, opts)
        return new_loop_vars
    set_state(iter_.reduce(init_vars, reduce_body))