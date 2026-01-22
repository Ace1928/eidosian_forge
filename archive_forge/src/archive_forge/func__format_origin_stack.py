import socket
import grpc
from tensorflow.core.debug import debug_service_pb2
from tensorflow.core.protobuf import debug_pb2
from tensorflow.python.debug.lib import common
from tensorflow.python.debug.lib import debug_service_pb2_grpc
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.platform import gfile
from tensorflow.python.profiler import tfprof_logger
def _format_origin_stack(origin_stack, call_traceback_proto):
    """Format a traceback stack for a `CallTraceback` proto.

  Args:
    origin_stack: The stack list as returned by `traceback.extract_stack()`.
    call_traceback_proto: A `CallTraceback` proto whose fields are to be
      populated.
  """
    string_to_id = {}
    string_to_id[None] = 0
    for frame in origin_stack:
        file_path, lineno, func_name, line_text = frame
        call_traceback_proto.origin_stack.traces.add(file_id=_string_to_id(file_path, string_to_id), lineno=lineno, function_id=_string_to_id(func_name, string_to_id), line_id=_string_to_id(line_text, string_to_id))
    id_to_string = call_traceback_proto.origin_id_to_string
    for key, value in string_to_id.items():
        id_to_string[value] = key if key is not None else ''