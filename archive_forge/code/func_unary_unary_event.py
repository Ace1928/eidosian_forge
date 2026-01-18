import collections
from grpc.framework.common import cardinality
from grpc.framework.common import style
from grpc.framework.foundation import stream  # pylint: disable=unused-import
from grpc.framework.interfaces.face import face
def unary_unary_event(behavior):
    """Creates an face.MethodImplementation for the given behavior.

    Args:
      behavior: The implementation of a unary-unary RPC method as a callable
        value that takes a request value, a response callback to which to pass
        the response value of the RPC, and an face.ServicerContext.

    Returns:
      An face.MethodImplementation derived from the given behavior.
    """
    return _MethodImplementation(cardinality.Cardinality.UNARY_UNARY, style.Service.EVENT, None, None, None, None, behavior, None, None, None)