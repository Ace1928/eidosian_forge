import collections
from grpc.framework.common import cardinality
from grpc.framework.common import style
from grpc.framework.foundation import stream  # pylint: disable=unused-import
from grpc.framework.interfaces.face import face
class _MethodImplementation(face.MethodImplementation, collections.namedtuple('_MethodImplementation', ['cardinality', 'style', 'unary_unary_inline', 'unary_stream_inline', 'stream_unary_inline', 'stream_stream_inline', 'unary_unary_event', 'unary_stream_event', 'stream_unary_event', 'stream_stream_event'])):
    pass