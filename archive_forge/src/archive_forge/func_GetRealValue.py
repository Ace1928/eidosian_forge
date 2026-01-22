from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import control_flow_v2_func_graphs
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
def GetRealValue(self, value):
    """Get the real value of `value`.

    If backprop "uses" a value produced by forward inference, an accumulator
    is added in the forward loop to accumulate its values.  We use the
    accumulated value. This method must be called in the grad loop context.
    `value` must be in forward and needed for backprop.

    Args:
      value: A tensor to be captured.

    Returns:
      The same tensor obtained from the saved history.
    """
    assert value.op.type not in ['Variable', 'VariableV2']
    real_value = self._history_map.get(value.name)
    if real_value is None:
        cur_value = value
        cur_grad_state = self
        while True:
            enter_op = util.GetLoopConstantEnter(cur_value)
            if enter_op:
                cur_value = enter_op.inputs[0]
                cur_grad_state = cur_grad_state.outer_grad_state
                if cur_grad_state is None:
                    real_value = self._grad_context.AddValue(cur_value)
                    break
            elif constant_op.is_constant(cur_value):
                real_value = constant_op.constant(tensor_util.constant_value(cur_value), dtype=cur_value.dtype)
                break
            else:
                self._grad_context.Exit()
                history_value = cur_grad_state.AddForwardAccumulator(cur_value)
                self._grad_context.Enter()
                break
        if real_value is None:
            real_value = cur_grad_state.AddBackpropAccumulatedValue(history_value, cur_value)
            if cur_grad_state != self:
                real_value = self._grad_context.AddValue(real_value)
        self._history_map[value.name] = real_value
    return real_value