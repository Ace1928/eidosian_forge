import grpc
from . import gcs_service_pb2 as src_dot_ray_dot_protobuf_dot_gcs__service__pb2
class NodeInfoGcsServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetClusterId = channel.unary_unary('/ray.rpc.NodeInfoGcsService/GetClusterId', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetClusterIdRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetClusterIdReply.FromString)
        self.RegisterNode = channel.unary_unary('/ray.rpc.NodeInfoGcsService/RegisterNode', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.RegisterNodeRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.RegisterNodeReply.FromString)
        self.DrainNode = channel.unary_unary('/ray.rpc.NodeInfoGcsService/DrainNode', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.DrainNodeRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.DrainNodeReply.FromString)
        self.GetAllNodeInfo = channel.unary_unary('/ray.rpc.NodeInfoGcsService/GetAllNodeInfo', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllNodeInfoRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllNodeInfoReply.FromString)
        self.GetInternalConfig = channel.unary_unary('/ray.rpc.NodeInfoGcsService/GetInternalConfig', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetInternalConfigRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetInternalConfigReply.FromString)
        self.CheckAlive = channel.unary_unary('/ray.rpc.NodeInfoGcsService/CheckAlive', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.CheckAliveRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.CheckAliveReply.FromString)