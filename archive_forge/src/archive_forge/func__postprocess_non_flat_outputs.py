import contextlib
from tensorflow.compiler.jit.ops import xla_ops
from tensorflow.compiler.jit.ops import xla_ops_grad  # pylint: disable=unused-import
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute import summary_op_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def _postprocess_non_flat_outputs(outputs):
    """Validates non-flat outputs and adds back device assignments.

  Args:
    outputs: Output from `computation` inside `xla.compile`.

  Returns:
    Tensors extracted from outputs and an empty list because Operations are not
    allowed in non-flat outputs..
  """
    new_output_tensors = []
    for o in nest.flatten(outputs):
        if isinstance(o, ops.Operation):
            raise ValueError('xla.compile does not support Operation as return value in non-flat output structure. You can set returned Operations as control dependencies of returned Tensors so Operations are triggered when Tensors are evaluated. Operation found: "%s"' % o.name)
        try:
            o = ops.convert_to_tensor(o)
        except Exception as e:
            raise ValueError('XLA computation function return values must all either be Operations or convertible to Tensors. Got error: "%s"' % str(e))
        with ops.device(o.device if o.device else ''):
            new_output_tensors.append(array_ops.identity(o))
    return (new_output_tensors, [])