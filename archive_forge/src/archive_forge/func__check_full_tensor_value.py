import numpy as np
from tensorflow.core.protobuf import debug_event_pb2
def _check_full_tensor_value(self, tensor_value, wall_time, op_type, output_slot, execution_index=None, graph_execution_trace_index=None):
    """Check a full tensor value.

    Appends to the list of alerts if any inf or nan is found in the full tensor
    value.

    Args:
      tensor_value: The full tensor value as a `np.ndarray`.
      wall_time: Wall timestamp for the execution event that generated the
        tensor value.
      op_type: Op type executed.
      output_slot: The output slot of the op.
      execution_index: Index to the top-level execution event.
      graph_execution_trace_index: Index to the intra-graph execution trace
        (if applicable.)
    """
    size = np.size(tensor_value)
    if not size or not np.issubdtype(tensor_value.dtype, np.floating):
        return
    is_inf = np.isinf(tensor_value)
    num_neg_inf = np.count_nonzero(np.logical_and(is_inf, np.less(tensor_value, 0.0)))
    num_pos_inf = np.count_nonzero(np.logical_and(is_inf, np.greater(tensor_value, 0.0)))
    num_nan = np.count_nonzero(np.isnan(tensor_value))
    if num_neg_inf or num_pos_inf or num_nan:
        self._alerts.append(InfNanAlert(wall_time, op_type, output_slot, size=size, num_neg_inf=num_neg_inf, num_pos_inf=num_pos_inf, num_nan=num_nan, execution_index=execution_index, graph_execution_trace_index=graph_execution_trace_index))