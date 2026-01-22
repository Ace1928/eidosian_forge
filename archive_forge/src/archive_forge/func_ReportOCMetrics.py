import grpc
from . import reporter_pb2 as src_dot_ray_dot_protobuf_dot_reporter__pb2
@staticmethod
def ReportOCMetrics(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
    return grpc.experimental.unary_unary(request, target, '/ray.rpc.ReporterService/ReportOCMetrics', src_dot_ray_dot_protobuf_dot_reporter__pb2.ReportOCMetricsRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_reporter__pb2.ReportOCMetricsReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)