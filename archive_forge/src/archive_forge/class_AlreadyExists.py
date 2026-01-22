from __future__ import absolute_import
from __future__ import unicode_literals
import http.client
from typing import Dict
from typing import Union
import warnings
from google.rpc import error_details_pb2
class AlreadyExists(Conflict):
    """Exception mapping a :attr:`grpc.StatusCode.ALREADY_EXISTS` error."""
    grpc_status_code = grpc.StatusCode.ALREADY_EXISTS if grpc is not None else None