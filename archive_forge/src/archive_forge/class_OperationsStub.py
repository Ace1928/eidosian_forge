import grpc
from google.longrunning import (
from cloudsdk.google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
class OperationsStub(object):
    """Manages long-running operations with an API service.

    When an API method normally takes long time to complete, it can be designed
    to return [Operation][google.longrunning.Operation] to the client, and the client can use this
    interface to receive the real response asynchronously by polling the
    operation resource, or pass the operation resource to another API (such as
    Google Cloud Pub/Sub API) to receive the response.  Any API service that
    returns long-running operations should implement the `Operations` interface
    so developers can have a consistent client experience.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ListOperations = channel.unary_unary('/google.longrunning.Operations/ListOperations', request_serializer=google_dot_longrunning_dot_operations__pb2.ListOperationsRequest.SerializeToString, response_deserializer=google_dot_longrunning_dot_operations__pb2.ListOperationsResponse.FromString)
        self.GetOperation = channel.unary_unary('/google.longrunning.Operations/GetOperation', request_serializer=google_dot_longrunning_dot_operations__pb2.GetOperationRequest.SerializeToString, response_deserializer=google_dot_longrunning_dot_operations__pb2.Operation.FromString)
        self.DeleteOperation = channel.unary_unary('/google.longrunning.Operations/DeleteOperation', request_serializer=google_dot_longrunning_dot_operations__pb2.DeleteOperationRequest.SerializeToString, response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString)
        self.CancelOperation = channel.unary_unary('/google.longrunning.Operations/CancelOperation', request_serializer=google_dot_longrunning_dot_operations__pb2.CancelOperationRequest.SerializeToString, response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString)
        self.WaitOperation = channel.unary_unary('/google.longrunning.Operations/WaitOperation', request_serializer=google_dot_longrunning_dot_operations__pb2.WaitOperationRequest.SerializeToString, response_deserializer=google_dot_longrunning_dot_operations__pb2.Operation.FromString)