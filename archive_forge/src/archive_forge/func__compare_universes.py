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
def _compare_universes(client_universe: str, credentials: ga_credentials.Credentials) -> bool:
    """Returns True iff the universe domains used by the client and credentials match.

        Args:
            client_universe (str): The universe domain configured via the client options.
            credentials (ga_credentials.Credentials): The credentials being used in the client.

        Returns:
            bool: True iff client_universe matches the universe in credentials.

        Raises:
            ValueError: when client_universe does not match the universe in credentials.
        """
    default_universe = SpeechClient._DEFAULT_UNIVERSE
    credentials_universe = getattr(credentials, 'universe_domain', default_universe)
    if client_universe != credentials_universe:
        raise ValueError(f"The configured universe domain ({client_universe}) does not match the universe domain found in the credentials ({credentials_universe}). If you haven't configured the universe domain explicitly, `{default_universe}` is the default.")
    return True