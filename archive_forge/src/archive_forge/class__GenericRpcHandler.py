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
class _GenericRpcHandler(grpc.GenericRpcHandler):

    def __init__(self, method_implementations, multi_method_implementation, request_deserializers, response_serializers):
        self._method_implementations = _flatten_method_pair_map(method_implementations)
        self._request_deserializers = _flatten_method_pair_map(request_deserializers)
        self._response_serializers = _flatten_method_pair_map(response_serializers)
        self._multi_method_implementation = multi_method_implementation

    def service(self, handler_call_details):
        method_implementation = self._method_implementations.get(handler_call_details.method)
        if method_implementation is not None:
            return _simple_method_handler(method_implementation, self._request_deserializers.get(handler_call_details.method), self._response_serializers.get(handler_call_details.method))
        elif self._multi_method_implementation is None:
            return None
        else:
            try:
                return None
            except face.NoSuchMethodError:
                return None