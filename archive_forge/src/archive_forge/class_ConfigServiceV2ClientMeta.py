from collections import OrderedDict
import os
import re
from typing import Dict, Mapping, MutableMapping, MutableSequence, Optional, Sequence, Tuple, Type, Union, cast
from googlecloudsdk.generated_clients.gapic_clients.logging_v2 import gapic_version as package_version
from google.api_core import client_options as client_options_lib
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials             # type: ignore
from google.auth.transport import mtls                            # type: ignore
from google.auth.transport.grpc import SslCredentials             # type: ignore
from google.auth.exceptions import MutualTLSChannelError          # type: ignore
from google.oauth2 import service_account                         # type: ignore
from google.api_core import operation  # type: ignore
from google.api_core import operation_async  # type: ignore
from cloudsdk.google.protobuf import empty_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from googlecloudsdk.generated_clients.gapic_clients.logging_v2.services.config_service_v2 import pagers
from googlecloudsdk.generated_clients.gapic_clients.logging_v2.types import logging_config
from .transports.base import ConfigServiceV2Transport, DEFAULT_CLIENT_INFO
from .transports.grpc import ConfigServiceV2GrpcTransport
from .transports.grpc_asyncio import ConfigServiceV2GrpcAsyncIOTransport
from .transports.rest import ConfigServiceV2RestTransport
class ConfigServiceV2ClientMeta(type):
    """Metaclass for the ConfigServiceV2 client.

    This provides class-level methods for building and retrieving
    support objects (e.g. transport) without polluting the client instance
    objects.
    """
    _transport_registry = OrderedDict()
    _transport_registry['grpc'] = ConfigServiceV2GrpcTransport
    _transport_registry['grpc_asyncio'] = ConfigServiceV2GrpcAsyncIOTransport
    _transport_registry['rest'] = ConfigServiceV2RestTransport

    def get_transport_class(cls, label: Optional[str]=None) -> Type[ConfigServiceV2Transport]:
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