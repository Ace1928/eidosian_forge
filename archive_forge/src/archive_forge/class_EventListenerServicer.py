import grpc
from tensorflow.core.debug import debug_service_pb2 as tensorflow_dot_core_dot_debug_dot_debug__service__pb2
from tensorflow.core.protobuf import debug_pb2 as tensorflow_dot_core_dot_protobuf_dot_debug__pb2
from tensorflow.core.util import event_pb2 as tensorflow_dot_core_dot_util_dot_event__pb2
class EventListenerServicer(object):
    """EventListener: Receives Event protos, e.g., from debugged TensorFlow
  runtime(s).
  """

    def SendEvents(self, request_iterator, context):
        """Client(s) can use this RPC method to send the EventListener Event protos.
    The Event protos can hold information such as:
    1) intermediate tensors from a debugged graph being executed, which can
    be sent from DebugIdentity ops configured with grpc URLs.
    2) GraphDefs of partition graphs, which can be sent from special debug
    ops that get executed immediately after the beginning of the graph
    execution.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendTracebacks(self, request, context):
        """Send the tracebacks of ops in a Python graph definition.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendSourceFiles(self, request, context):
        """Send a collection of source code files being debugged.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')