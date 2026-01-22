import functools
import threading
from google.api_core import exceptions
from google.api_core import protobuf_helpers
from google.api_core.future import polling
from google.longrunning import operations_pb2
from cloudsdk.google.protobuf import json_format
from google.rpc import code_pb2
def _refresh_grpc(operations_stub, operation_name, retry=None):
    """Refresh an operation using a gRPC client.

    Args:
        operations_stub (google.longrunning.operations_pb2.OperationsStub):
            The gRPC operations stub.
        operation_name (str): The name of the operation.
        retry (google.api_core.retry.Retry): (Optional) retry policy

    Returns:
        google.longrunning.operations_pb2.Operation: The operation.
    """
    request_pb = operations_pb2.GetOperationRequest(name=operation_name)
    rpc = operations_stub.GetOperation
    if retry is not None:
        rpc = retry(rpc)
    return rpc(request_pb)