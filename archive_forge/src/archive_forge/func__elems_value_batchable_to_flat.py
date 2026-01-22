import re
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _elems_value_batchable_to_flat(elems_value_batchable, elems_flat_signature):
    """Converts elems_value_batchable -> elems_value_flat."""
    elems_value_flat = []
    i = 0
    for spec in elems_flat_signature:
        spec = spec._unbatch()
        tensor_list = elems_value_batchable[i:i + len(spec._flat_tensor_specs)]
        elems_value_flat.append(spec._from_compatible_tensor_list(tensor_list))
        i += len(tensor_list)
    assert i == len(elems_value_batchable)
    return elems_value_flat