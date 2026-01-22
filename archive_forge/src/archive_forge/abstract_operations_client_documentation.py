from collections import OrderedDict
import os
import re
from typing import Dict, Optional, Sequence, Tuple, Type, Union
from google.api_core import client_options as client_options_lib  # type: ignore
from google.api_core import gapic_v1  # type: ignore
from google.api_core import retry as retries  # type: ignore
from google.api_core.operations_v1 import pagers
from google.api_core.operations_v1.transports.base import (
from google.api_core.operations_v1.transports.rest import OperationsRestTransport
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.exceptions import MutualTLSChannelError  # type: ignore
from google.auth.transport import mtls  # type: ignore
from google.longrunning import operations_pb2
from google.oauth2 import service_account  # type: ignore
import grpc
Starts asynchronous cancellation on a long-running operation.
        The server makes a best effort to cancel the operation, but
        success is not guaranteed. If the server doesn't support this
        method, it returns ``google.rpc.Code.UNIMPLEMENTED``. Clients
        can use
        [Operations.GetOperation][google.api_core.operations_v1.Operations.GetOperation]
        or other methods to check whether the cancellation succeeded or
        whether the operation completed despite cancellation. On
        successful cancellation, the operation is not deleted; instead,
        it becomes an operation with an
        [Operation.error][google.api_core.operations_v1.Operation.error] value with
        a [google.rpc.Status.code][google.rpc.Status.code] of 1,
        corresponding to ``Code.CANCELLED``.

        Args:
            name (str):
                The name of the operation resource to
                be cancelled.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        