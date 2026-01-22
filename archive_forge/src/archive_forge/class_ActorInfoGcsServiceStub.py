import grpc
from . import gcs_service_pb2 as src_dot_ray_dot_protobuf_dot_gcs__service__pb2
class ActorInfoGcsServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.RegisterActor = channel.unary_unary('/ray.rpc.ActorInfoGcsService/RegisterActor', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.RegisterActorRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.RegisterActorReply.FromString)
        self.CreateActor = channel.unary_unary('/ray.rpc.ActorInfoGcsService/CreateActor', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.CreateActorRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.CreateActorReply.FromString)
        self.GetActorInfo = channel.unary_unary('/ray.rpc.ActorInfoGcsService/GetActorInfo', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetActorInfoRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetActorInfoReply.FromString)
        self.GetNamedActorInfo = channel.unary_unary('/ray.rpc.ActorInfoGcsService/GetNamedActorInfo', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetNamedActorInfoRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetNamedActorInfoReply.FromString)
        self.ListNamedActors = channel.unary_unary('/ray.rpc.ActorInfoGcsService/ListNamedActors', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.ListNamedActorsRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.ListNamedActorsReply.FromString)
        self.GetAllActorInfo = channel.unary_unary('/ray.rpc.ActorInfoGcsService/GetAllActorInfo', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllActorInfoRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllActorInfoReply.FromString)
        self.KillActorViaGcs = channel.unary_unary('/ray.rpc.ActorInfoGcsService/KillActorViaGcs', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.KillActorViaGcsRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.KillActorViaGcsReply.FromString)