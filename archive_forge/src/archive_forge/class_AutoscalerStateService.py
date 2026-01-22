import grpc
from . import autoscaler_pb2 as src_dot_ray_dot_protobuf_dot_autoscaler__pb2
class AutoscalerStateService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetClusterResourceState(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.autoscaler.AutoscalerStateService/GetClusterResourceState', src_dot_ray_dot_protobuf_dot_autoscaler__pb2.GetClusterResourceStateRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_autoscaler__pb2.GetClusterResourceStateReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ReportAutoscalingState(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.autoscaler.AutoscalerStateService/ReportAutoscalingState', src_dot_ray_dot_protobuf_dot_autoscaler__pb2.ReportAutoscalingStateRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_autoscaler__pb2.ReportAutoscalingStateReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RequestClusterResourceConstraint(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.autoscaler.AutoscalerStateService/RequestClusterResourceConstraint', src_dot_ray_dot_protobuf_dot_autoscaler__pb2.RequestClusterResourceConstraintRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_autoscaler__pb2.RequestClusterResourceConstraintReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetClusterStatus(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.autoscaler.AutoscalerStateService/GetClusterStatus', src_dot_ray_dot_protobuf_dot_autoscaler__pb2.GetClusterStatusRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_autoscaler__pb2.GetClusterStatusReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DrainNode(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.autoscaler.AutoscalerStateService/DrainNode', src_dot_ray_dot_protobuf_dot_autoscaler__pb2.DrainNodeRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_autoscaler__pb2.DrainNodeReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)