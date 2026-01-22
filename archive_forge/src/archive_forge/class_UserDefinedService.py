import grpc
from . import serve_pb2 as src_dot_ray_dot_protobuf_dot_serve__pb2
class UserDefinedService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def __call__(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.serve.UserDefinedService/__call__', src_dot_ray_dot_protobuf_dot_serve__pb2.UserDefinedMessage.SerializeToString, src_dot_ray_dot_protobuf_dot_serve__pb2.UserDefinedResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Method1(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.serve.UserDefinedService/Method1', src_dot_ray_dot_protobuf_dot_serve__pb2.UserDefinedMessage.SerializeToString, src_dot_ray_dot_protobuf_dot_serve__pb2.UserDefinedResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Method2(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.serve.UserDefinedService/Method2', src_dot_ray_dot_protobuf_dot_serve__pb2.UserDefinedMessage2.SerializeToString, src_dot_ray_dot_protobuf_dot_serve__pb2.UserDefinedResponse2.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Streaming(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_stream(request, target, '/ray.serve.UserDefinedService/Streaming', src_dot_ray_dot_protobuf_dot_serve__pb2.UserDefinedMessage.SerializeToString, src_dot_ray_dot_protobuf_dot_serve__pb2.UserDefinedResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)