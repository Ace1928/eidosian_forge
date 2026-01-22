import collections
import threading
import grpc
from grpc import _common
from grpc.beta import _metadata
from grpc.beta import interfaces
from grpc.framework.common import cardinality
from grpc.framework.common import style
from grpc.framework.foundation import abandonment
from grpc.framework.foundation import logging_pool
from grpc.framework.foundation import stream
from grpc.framework.interfaces.face import face
def _adapt_stream_request_inline(stream_request_inline):

    def adaptation(request_iterator, servicer_context):
        return stream_request_inline(request_iterator, _FaceServicerContext(servicer_context))
    return adaptation