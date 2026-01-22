from collections import OrderedDict
import functools
import os
import re
from typing import (
from google.pubsub_v1 import gapic_version as package_version
from google.api_core import client_options as client_options_lib
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.api_core import timeout as timeouts  # type: ignore
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport import mtls  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.auth.exceptions import MutualTLSChannelError  # type: ignore
from google.oauth2 import service_account  # type: ignore
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from google.pubsub_v1.services.publisher import pagers
from google.pubsub_v1.types import pubsub
from google.pubsub_v1.types import TimeoutType
import grpc
from .transports.base import PublisherTransport, DEFAULT_CLIENT_INFO
from .transports.grpc import PublisherGrpcTransport
from .transports.grpc_asyncio import PublisherGrpcAsyncIOTransport
from .transports.rest import PublisherRestTransport
class PublisherClientMeta(type):
    """Metaclass for the Publisher client.

    This provides class-level methods for building and retrieving
    support objects (e.g. transport) without polluting the client instance
    objects.
    """
    _transport_registry = OrderedDict()
    _transport_registry['grpc'] = PublisherGrpcTransport
    _transport_registry['grpc_asyncio'] = PublisherGrpcAsyncIOTransport
    _transport_registry['rest'] = PublisherRestTransport

    def get_transport_class(cls, label: Optional[str]=None) -> Type[PublisherTransport]:
        """Returns an appropriate transport class.

        Args:
            label: The name of the desired transport. If none is
                provided, then the first transport in the registry is used.

        Returns:
            The transport class to use.
        """
        if label:
            return cls._transport_registry[label]
        return next(iter(cls._transport_registry.values()))