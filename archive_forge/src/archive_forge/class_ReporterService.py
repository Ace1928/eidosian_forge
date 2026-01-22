import grpc
from . import reporter_pb2 as src_dot_ray_dot_protobuf_dot_reporter__pb2
class ReporterService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetProfilingStats(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.ReporterService/GetProfilingStats', src_dot_ray_dot_protobuf_dot_reporter__pb2.GetProfilingStatsRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_reporter__pb2.GetProfilingStatsReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ReportMetrics(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.ReporterService/ReportMetrics', src_dot_ray_dot_protobuf_dot_reporter__pb2.ReportMetricsRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_reporter__pb2.ReportMetricsReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ReportOCMetrics(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.ReporterService/ReportOCMetrics', src_dot_ray_dot_protobuf_dot_reporter__pb2.ReportOCMetricsRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_reporter__pb2.ReportOCMetricsReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetTraceback(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.ReporterService/GetTraceback', src_dot_ray_dot_protobuf_dot_reporter__pb2.GetTracebackRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_reporter__pb2.GetTracebackReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CpuProfiling(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.ReporterService/CpuProfiling', src_dot_ray_dot_protobuf_dot_reporter__pb2.CpuProfilingRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_reporter__pb2.CpuProfilingReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)