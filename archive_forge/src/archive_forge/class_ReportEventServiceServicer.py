import grpc
from . import event_pb2 as src_dot_ray_dot_protobuf_dot_event__pb2
class ReportEventServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ReportEvents(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')