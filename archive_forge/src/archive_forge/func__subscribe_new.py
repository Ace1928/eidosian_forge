import contextlib
import re
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
def _subscribe_new(tensor, side_effects, control_cache):
    """Helper method that subscribes a single tensor to a list of side_effects.

  Args:
    tensor: `tf.Tensor`
    side_effects: List of side_effect functions see subscribe for details.
    control_cache: `_ControlOutputCache` helper to get control_outputs faster.

  Returns:
    The modified replacement to the passed in tensor which triggers the side
    effects.
  """
    update_input = []
    for consumer_op in list(tensor.consumers()):
        update_input.append((consumer_op, list(consumer_op.inputs).index(tensor)))
    update_control_input = control_cache.get_control_outputs(tensor.op)
    name_scope = tensor.op.name + '/subscription/'
    with ops.name_scope(name_scope):
        outs = []
        for s in side_effects:
            outs += s(tensor)
        with ops.control_dependencies(outs):
            out = array_ops.identity(tensor)
    for consumer_op, index in update_input:
        consumer_op._update_input(index, out)
    for consumer_op in update_control_input:
        new_control_inputs = consumer_op.control_inputs
        if tensor.op in new_control_inputs:
            new_control_inputs.remove(tensor.op)
        new_control_inputs.append(out.op)
        consumer_op._remove_all_control_inputs()
        consumer_op._add_control_inputs(new_control_inputs)
    return out