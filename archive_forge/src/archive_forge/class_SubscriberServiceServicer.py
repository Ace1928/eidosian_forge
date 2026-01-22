import grpc
from . import pubsub_pb2 as src_dot_ray_dot_protobuf_dot_pubsub__pb2
class SubscriberServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def PubsubLongPolling(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def PubsubCommandBatch(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')