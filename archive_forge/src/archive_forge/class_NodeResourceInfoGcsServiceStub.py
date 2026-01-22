import grpc
from . import gcs_service_pb2 as src_dot_ray_dot_protobuf_dot_gcs__service__pb2
class NodeResourceInfoGcsServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetAllAvailableResources = channel.unary_unary('/ray.rpc.NodeResourceInfoGcsService/GetAllAvailableResources', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllAvailableResourcesRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllAvailableResourcesReply.FromString)
        self.GetAllResourceUsage = channel.unary_unary('/ray.rpc.NodeResourceInfoGcsService/GetAllResourceUsage', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllResourceUsageRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllResourceUsageReply.FromString)
        self.GetDrainingNodes = channel.unary_unary('/ray.rpc.NodeResourceInfoGcsService/GetDrainingNodes', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetDrainingNodesRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetDrainingNodesReply.FromString)