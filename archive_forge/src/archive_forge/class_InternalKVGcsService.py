import grpc
from . import gcs_service_pb2 as src_dot_ray_dot_protobuf_dot_gcs__service__pb2
class InternalKVGcsService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def InternalKVGet(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.InternalKVGcsService/InternalKVGet', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.InternalKVGetRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.InternalKVGetReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def InternalKVMultiGet(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.InternalKVGcsService/InternalKVMultiGet', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.InternalKVMultiGetRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.InternalKVMultiGetReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def InternalKVPut(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.InternalKVGcsService/InternalKVPut', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.InternalKVPutRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.InternalKVPutReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def InternalKVDel(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.InternalKVGcsService/InternalKVDel', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.InternalKVDelRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.InternalKVDelReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def InternalKVExists(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.InternalKVGcsService/InternalKVExists', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.InternalKVExistsRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.InternalKVExistsReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def InternalKVKeys(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.InternalKVGcsService/InternalKVKeys', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.InternalKVKeysRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.InternalKVKeysReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)