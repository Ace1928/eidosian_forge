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
def _adapt_unary_stream_event(unary_stream_event):

    def adaptation(request, servicer_context):
        callback = _Callback()
        if not servicer_context.add_callback(callback.cancel):
            raise abandonment.Abandoned()
        unary_stream_event(request, callback, _FaceServicerContext(servicer_context))
        while True:
            response = callback.draw_one_value()
            if response is None:
                return
            else:
                yield response
    return adaptation