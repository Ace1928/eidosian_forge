import grpc
from . import ray_client_pb2 as src_dot_ray_dot_protobuf_dot_ray__client__pb2
class RayletDriverStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Init = channel.unary_unary('/ray.rpc.RayletDriver/Init', request_serializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.InitRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.InitResponse.FromString)
        self.PrepRuntimeEnv = channel.unary_unary('/ray.rpc.RayletDriver/PrepRuntimeEnv', request_serializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.PrepRuntimeEnvRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.PrepRuntimeEnvResponse.FromString)
        self.GetObject = channel.unary_stream('/ray.rpc.RayletDriver/GetObject', request_serializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.GetRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.GetResponse.FromString)
        self.PutObject = channel.unary_unary('/ray.rpc.RayletDriver/PutObject', request_serializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.PutRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.PutResponse.FromString)
        self.WaitObject = channel.unary_unary('/ray.rpc.RayletDriver/WaitObject', request_serializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.WaitRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.WaitResponse.FromString)
        self.Schedule = channel.unary_unary('/ray.rpc.RayletDriver/Schedule', request_serializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.ClientTask.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.ClientTaskTicket.FromString)
        self.Terminate = channel.unary_unary('/ray.rpc.RayletDriver/Terminate', request_serializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.TerminateRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.TerminateResponse.FromString)
        self.ClusterInfo = channel.unary_unary('/ray.rpc.RayletDriver/ClusterInfo', request_serializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.ClusterInfoRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.ClusterInfoResponse.FromString)
        self.KVGet = channel.unary_unary('/ray.rpc.RayletDriver/KVGet', request_serializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.KVGetRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.KVGetResponse.FromString)
        self.KVPut = channel.unary_unary('/ray.rpc.RayletDriver/KVPut', request_serializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.KVPutRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.KVPutResponse.FromString)
        self.KVDel = channel.unary_unary('/ray.rpc.RayletDriver/KVDel', request_serializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.KVDelRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.KVDelResponse.FromString)
        self.KVList = channel.unary_unary('/ray.rpc.RayletDriver/KVList', request_serializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.KVListRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.KVListResponse.FromString)
        self.KVExists = channel.unary_unary('/ray.rpc.RayletDriver/KVExists', request_serializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.KVExistsRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.KVExistsResponse.FromString)
        self.ListNamedActors = channel.unary_unary('/ray.rpc.RayletDriver/ListNamedActors', request_serializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.ClientListNamedActorsRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.ClientListNamedActorsResponse.FromString)
        self.PinRuntimeEnvURI = channel.unary_unary('/ray.rpc.RayletDriver/PinRuntimeEnvURI', request_serializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.ClientPinRuntimeEnvURIRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_ray__client__pb2.ClientPinRuntimeEnvURIResponse.FromString)