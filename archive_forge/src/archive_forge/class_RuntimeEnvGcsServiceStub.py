import grpc
from . import gcs_service_pb2 as src_dot_ray_dot_protobuf_dot_gcs__service__pb2
class RuntimeEnvGcsServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.PinRuntimeEnvURI = channel.unary_unary('/ray.rpc.RuntimeEnvGcsService/PinRuntimeEnvURI', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.PinRuntimeEnvURIRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.PinRuntimeEnvURIReply.FromString)