import grpc
from grpc import _common
from grpc.beta import _metadata
from grpc.beta import interfaces
from grpc.framework.common import cardinality
from grpc.framework.foundation import future
from grpc.framework.interfaces.face import face
class _InvocationProtocolContext(interfaces.GRPCInvocationContext):

    def disable_next_request_compression(self):
        pass