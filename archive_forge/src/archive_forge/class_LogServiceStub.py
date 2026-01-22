import grpc
from . import reporter_pb2 as src_dot_ray_dot_protobuf_dot_reporter__pb2
class LogServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ListLogs = channel.unary_unary('/ray.rpc.LogService/ListLogs', request_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.ListLogsRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.ListLogsReply.FromString)
        self.StreamLog = channel.unary_stream('/ray.rpc.LogService/StreamLog', request_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.StreamLogRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.StreamLogReply.FromString)