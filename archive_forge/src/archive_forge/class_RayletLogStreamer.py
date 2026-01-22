import grpc
from . import ray_client_pb2 as src_dot_ray_dot_protobuf_dot_ray__client__pb2
class RayletLogStreamer(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Logstream(request_iterator, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/ray.rpc.RayletLogStreamer/Logstream', src_dot_ray_dot_protobuf_dot_ray__client__pb2.LogSettingsRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_ray__client__pb2.LogData.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)