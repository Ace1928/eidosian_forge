import grpc
from grpc import _common
from grpc.beta import _metadata
from grpc.beta import interfaces
from grpc.framework.common import cardinality
from grpc.framework.foundation import future
from grpc.framework.interfaces.face import face
def _unary_stream(channel, group, method, timeout, protocol_options, metadata, metadata_transformer, request, request_serializer, response_deserializer):
    multi_callable = channel.unary_stream(_common.fully_qualified_method(group, method), request_serializer=request_serializer, response_deserializer=response_deserializer)
    effective_metadata = _effective_metadata(metadata, metadata_transformer)
    response_iterator = multi_callable(request, timeout=timeout, metadata=_metadata.unbeta(effective_metadata), credentials=_credentials(protocol_options))
    return _Rendezvous(None, response_iterator, response_iterator)