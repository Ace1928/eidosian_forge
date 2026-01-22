import grpc
from . import event_pb2 as src_dot_ray_dot_protobuf_dot_event__pb2
class ReportEventServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ReportEvents = channel.unary_unary('/ray.rpc.ReportEventService/ReportEvents', request_serializer=src_dot_ray_dot_protobuf_dot_event__pb2.ReportEventsRequest.SerializeToString, response_deserializer=src_dot_ray_dot_protobuf_dot_event__pb2.ReportEventsReply.FromString)