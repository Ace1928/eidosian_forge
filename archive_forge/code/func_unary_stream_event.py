import collections
from grpc.framework.common import cardinality
from grpc.framework.common import style
from grpc.framework.foundation import stream  # pylint: disable=unused-import
from grpc.framework.interfaces.face import face
def unary_stream_event(behavior):
    """Creates an face.MethodImplementation for the given behavior.

    Args:
      behavior: The implementation of a unary-stream RPC method as a callable
        value that takes a request value, a stream.Consumer to which to pass the
        the response values of the RPC, and an face.ServicerContext.

    Returns:
      An face.MethodImplementation derived from the given behavior.
    """
    return _MethodImplementation(cardinality.Cardinality.UNARY_STREAM, style.Service.EVENT, None, None, None, None, None, behavior, None, None)