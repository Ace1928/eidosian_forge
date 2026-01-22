import grpc
from . import gcs_service_pb2 as src_dot_ray_dot_protobuf_dot_gcs__service__pb2
class WorkerInfoGcsServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ReportWorkerFailure = channel.unary_unary('/ray.rpc.WorkerInfoGcsService/ReportWorkerFailure', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.ReportWorkerFailureRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.ReportWorkerFailureReply.FromString)
        self.GetWorkerInfo = channel.unary_unary('/ray.rpc.WorkerInfoGcsService/GetWorkerInfo', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetWorkerInfoRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetWorkerInfoReply.FromString)
        self.GetAllWorkerInfo = channel.unary_unary('/ray.rpc.WorkerInfoGcsService/GetAllWorkerInfo', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllWorkerInfoRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.GetAllWorkerInfoReply.FromString)
        self.AddWorkerInfo = channel.unary_unary('/ray.rpc.WorkerInfoGcsService/AddWorkerInfo', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.AddWorkerInfoRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.AddWorkerInfoReply.FromString)
        self.UpdateWorkerDebuggerPort = channel.unary_unary('/ray.rpc.WorkerInfoGcsService/UpdateWorkerDebuggerPort', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.UpdateWorkerDebuggerPortRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.UpdateWorkerDebuggerPortReply.FromString)
        self.UpdateWorkerNumPausedThreads = channel.unary_unary('/ray.rpc.WorkerInfoGcsService/UpdateWorkerNumPausedThreads', request_serializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.UpdateWorkerNumPausedThreadsRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_gcs__service__pb2.UpdateWorkerNumPausedThreadsReply.FromString)