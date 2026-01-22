from __future__ import absolute_import
from __future__ import unicode_literals
import http.client
from typing import Dict
from typing import Union
import warnings
from google.rpc import error_details_pb2
class Cancelled(ClientError):
    """Exception mapping a :attr:`grpc.StatusCode.CANCELLED` error."""
    code = 499
    grpc_status_code = grpc.StatusCode.CANCELLED if grpc is not None else None