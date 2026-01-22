import grpc
from . import serve_pb2 as src_dot_ray_dot_protobuf_dot_serve__pb2
class RayServeBenchmarkServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.grpc_call = channel.unary_unary('/ray.serve.RayServeBenchmarkService/grpc_call', request_serializer=src_dot_ray_dot_protobuf_dot_serve__pb2.RawData.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_serve__pb2.ModelOutput.FromString)