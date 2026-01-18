import collections
import sys
from google.rpc import status_pb2
import grpc
from ._common import GRPC_DETAILS_METADATA_KEY
from ._common import code_to_grpc_status_code
def to_status(status):
    """Convert a google.rpc.status.Status message to grpc.Status.

    This is an EXPERIMENTAL API.

    Args:
      status: a google.rpc.status.Status message representing the non-OK status
        to terminate the RPC with and communicate it to the client.

    Returns:
      A grpc.Status instance representing the input google.rpc.status.Status message.
    """
    return _Status(code=code_to_grpc_status_code(status.code), details=status.message, trailing_metadata=((GRPC_DETAILS_METADATA_KEY, status.SerializeToString()),))