import grpc
from grpc import _common
from grpc.beta import _metadata
from grpc.beta import interfaces
from grpc.framework.common import cardinality
from grpc.framework.foundation import future
from grpc.framework.interfaces.face import face
class _StreamStreamMultiCallable(face.StreamStreamMultiCallable):

    def __init__(self, channel, group, method, metadata_transformer, request_serializer, response_deserializer):
        self._channel = channel
        self._group = group
        self._method = method
        self._metadata_transformer = metadata_transformer
        self._request_serializer = request_serializer
        self._response_deserializer = response_deserializer

    def __call__(self, request_iterator, timeout, metadata=None, protocol_options=None):
        return _stream_stream(self._channel, self._group, self._method, timeout, protocol_options, metadata, self._metadata_transformer, request_iterator, self._request_serializer, self._response_deserializer)

    def event(self, receiver, abortion_callback, timeout, metadata=None, protocol_options=None):
        raise NotImplementedError()