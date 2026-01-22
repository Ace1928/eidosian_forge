import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def _execution_from_debug_event_proto(debug_event, locator):
    """Convert a DebugEvent proto into an Execution data object."""
    execution_proto = debug_event.execution
    debug_tensor_values = None
    if execution_proto.tensor_debug_mode == debug_event_pb2.TensorDebugMode.FULL_TENSOR:
        pass
    elif execution_proto.tensor_debug_mode != debug_event_pb2.TensorDebugMode.NO_TENSOR:
        debug_tensor_values = []
        for tensor_proto in execution_proto.tensor_protos:
            debug_tensor_values.append(_parse_tensor_value(tensor_proto, return_list=True))
    return Execution(_execution_digest_from_debug_event_proto(debug_event, locator), execution_proto.code_location.host_name, tuple(execution_proto.code_location.stack_frame_ids), execution_proto.tensor_debug_mode, graph_id=execution_proto.graph_id, input_tensor_ids=tuple(execution_proto.input_tensor_ids), output_tensor_ids=tuple(execution_proto.output_tensor_ids), debug_tensor_values=_tuple_or_none(debug_tensor_values))