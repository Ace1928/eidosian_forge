import grpc
from . import gcs_service_pb2 as src_dot_ray_dot_protobuf_dot_gcs__service__pb2
class PlacementGroupInfoGcsService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def CreatePlacementGroup(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.PlacementGroupInfoGcsService/CreatePlacementGroup', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.CreatePlacementGroupRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.CreatePlacementGroupReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RemovePlacementGroup(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.PlacementGroupInfoGcsService/RemovePlacementGroup', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.RemovePlacementGroupRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.RemovePlacementGroupReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetPlacementGroup(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.PlacementGroupInfoGcsService/GetPlacementGroup', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetPlacementGroupRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetPlacementGroupReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetNamedPlacementGroup(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.PlacementGroupInfoGcsService/GetNamedPlacementGroup', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetNamedPlacementGroupRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetNamedPlacementGroupReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetAllPlacementGroup(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.PlacementGroupInfoGcsService/GetAllPlacementGroup', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllPlacementGroupRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllPlacementGroupReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def WaitPlacementGroupUntilReady(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.PlacementGroupInfoGcsService/WaitPlacementGroupUntilReady', src_dot_ray_dot_protobuf_dot_gcs__service__pb2.WaitPlacementGroupUntilReadyRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_gcs__service__pb2.WaitPlacementGroupUntilReadyReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)