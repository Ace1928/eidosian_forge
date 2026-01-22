import numpy as np
from tensorflow.core.protobuf import debug_event_pb2
def _check_debug_tensor_value(self, tensor_debug_mode, debug_tensor_value, wall_time, op_type, output_slot, execution_index=None, graph_execution_trace_index=None):
    """Check for bad numerical values based on debug summary of tensor value.

    If tensor_debug_mode is one in which debug_tensor_value does not carry
    information about the presence or count of inf / nan values (e.g., SHAPE),
    this method is a no-op.

    When infs and/or nans are found, `InfNanAlert` objects are created and
    appended to `self._alerts`.

    Args:
      tensor_debug_mode: TensorDebugMode proto enum.
      debug_tensor_value: Debug tensor value as a list of numbers.
      wall_time: Wall timestamp for the tensor event.
      op_type: Type of the op that generated the tensor (e.g., "Conv2D").
      output_slot: Output slot index of the tensor for the op.
      execution_index: Top-level execution index.
      graph_execution_trace_index: Intra-graph execution index.
    """
    assert tensor_debug_mode != debug_event_pb2.TensorDebugMode.FULL_TENSOR
    if not debug_tensor_value:
        return
    if tensor_debug_mode == debug_event_pb2.TensorDebugMode.CURT_HEALTH:
        _, any_nan_inf = debug_tensor_value
        if any_nan_inf:
            self._alerts.append(InfNanAlert(wall_time, op_type, output_slot, execution_index=execution_index, graph_execution_trace_index=graph_execution_trace_index))
    elif tensor_debug_mode == debug_event_pb2.TensorDebugMode.CONCISE_HEALTH:
        _, size, num_neg_inf, num_pos_inf, num_nan = debug_tensor_value
        if num_neg_inf or num_pos_inf or num_nan:
            self._alerts.append(InfNanAlert(wall_time, op_type, output_slot, size=size, num_neg_inf=num_neg_inf, num_pos_inf=num_pos_inf, num_nan=num_nan, execution_index=execution_index, graph_execution_trace_index=graph_execution_trace_index))
    elif tensor_debug_mode == debug_event_pb2.TensorDebugMode.FULL_HEALTH:
        _, _, _, _, size, num_neg_inf, num_pos_inf, num_nan, _, _, _ = debug_tensor_value
        if num_neg_inf or num_pos_inf or num_nan:
            self._alerts.append(InfNanAlert(wall_time, op_type, output_slot, size=size, num_neg_inf=num_neg_inf, num_pos_inf=num_pos_inf, num_nan=num_nan, execution_index=execution_index, graph_execution_trace_index=graph_execution_trace_index))