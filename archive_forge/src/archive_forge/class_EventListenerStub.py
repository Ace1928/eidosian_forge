import grpc
from tensorflow.core.debug import debug_service_pb2 as tensorflow_dot_core_dot_debug_dot_debug__service__pb2
from tensorflow.core.protobuf import debug_pb2 as tensorflow_dot_core_dot_protobuf_dot_debug__pb2
from tensorflow.core.util import event_pb2 as tensorflow_dot_core_dot_util_dot_event__pb2
class EventListenerStub(object):
    """EventListener: Receives Event protos, e.g., from debugged TensorFlow
  runtime(s).
  """

    def __init__(self, channel):
        """Constructor.

    Args:
      channel: A grpc.Channel.
    """
        self.SendEvents = channel.stream_stream('/tensorflow.EventListener/SendEvents', request_serializer=tensorflow_dot_core_dot_util_dot_event__pb2.Event.SerializeToString, response_deserializer=tensorflow_dot_core_dot_debug_dot_debug__service__pb2.EventReply.FromString)
        self.SendTracebacks = channel.unary_unary('/tensorflow.EventListener/SendTracebacks', request_serializer=tensorflow_dot_core_dot_debug_dot_debug__service__pb2.CallTraceback.SerializeToString, response_deserializer=tensorflow_dot_core_dot_debug_dot_debug__service__pb2.EventReply.FromString)
        self.SendSourceFiles = channel.unary_unary('/tensorflow.EventListener/SendSourceFiles', request_serializer=tensorflow_dot_core_dot_protobuf_dot_debug__pb2.DebuggedSourceFiles.SerializeToString, response_deserializer=tensorflow_dot_core_dot_debug_dot_debug__service__pb2.EventReply.FromString)