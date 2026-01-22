from collections import OrderedDict
import os
import re
from typing import (
import warnings
from google.api_core import client_options as client_options_lib
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.exceptions import MutualTLSChannelError  # type: ignore
from google.auth.transport import mtls  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.oauth2 import service_account  # type: ignore
from google.cloud.speech_v1p1beta1 import gapic_version as package_version
from google.api_core import operation  # type: ignore
from google.api_core import operation_async  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import duration_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
from google.cloud.speech_v1p1beta1.types import cloud_speech
from .transports.base import DEFAULT_CLIENT_INFO, SpeechTransport
from .transports.grpc import SpeechGrpcTransport
from .transports.grpc_asyncio import SpeechGrpcAsyncIOTransport
from .transports.rest import SpeechRestTransport
@staticmethod
def _get_client_cert_source(provided_cert_source, use_cert_flag):
    """Return the client cert source to be used by the client.

        Args:
            provided_cert_source (bytes): The client certificate source provided.
            use_cert_flag (bool): A flag indicating whether to use the client certificate.

        Returns:
            bytes or None: The client cert source to be used by the client.
        """
    client_cert_source = None
    if use_cert_flag:
        if provided_cert_source:
            client_cert_source = provided_cert_source
        elif mtls.has_default_client_cert_source():
            client_cert_source = mtls.default_client_cert_source()
    return client_cert_source