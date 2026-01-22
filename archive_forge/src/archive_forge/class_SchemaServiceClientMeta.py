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
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport import mtls  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.auth.exceptions import MutualTLSChannelError  # type: ignore
from google.oauth2 import service_account  # type: ignore
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.pubsub_v1.services.schema_service import pagers
from google.pubsub_v1.types import schema
from google.pubsub_v1.types import schema as gp_schema
import grpc
from .transports.base import SchemaServiceTransport, DEFAULT_CLIENT_INFO
from .transports.grpc import SchemaServiceGrpcTransport
from .transports.grpc_asyncio import SchemaServiceGrpcAsyncIOTransport
from .transports.rest import SchemaServiceRestTransport
class SchemaServiceClientMeta(type):
    """Metaclass for the SchemaService client.

    This provides class-level methods for building and retrieving
    support objects (e.g. transport) without polluting the client instance
    objects.
    """
    _transport_registry = OrderedDict()
    _transport_registry['grpc'] = SchemaServiceGrpcTransport
    _transport_registry['grpc_asyncio'] = SchemaServiceGrpcAsyncIOTransport
    _transport_registry['rest'] = SchemaServiceRestTransport

    def get_transport_class(cls, label: Optional[str]=None) -> Type[SchemaServiceTransport]:
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