import grpc
from . import reporter_pb2 as src_dot_ray_dot_protobuf_dot_reporter__pb2
@staticmethod
def CpuProfiling(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
    return grpc.experimental.unary_unary(request, target, '/ray.rpc.ReporterService/CpuProfiling', src_dot_ray_dot_protobuf_dot_reporter__pb2.CpuProfilingRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_reporter__pb2.CpuProfilingReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)