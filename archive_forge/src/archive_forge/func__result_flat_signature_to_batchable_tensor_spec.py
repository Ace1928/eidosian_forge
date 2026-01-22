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
def _result_flat_signature_to_batchable_tensor_spec(result_flat_signature):
    """Converts result_flat_signature -> result_batchable_tensor_specs."""
    tensor_specs = []
    for spec in result_flat_signature:
        if not isinstance(spec, type_spec.BatchableTypeSpec):
            raise TypeError('map_fn can not generate %s outputs' % (spec,))
        tensor_specs.extend(spec._flat_tensor_specs)
    return tensor_specs