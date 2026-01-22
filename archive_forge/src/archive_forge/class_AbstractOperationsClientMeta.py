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
class AbstractOperationsClientMeta(type):
    """Metaclass for the Operations client.

    This provides class-level methods for building and retrieving
    support objects (e.g. transport) without polluting the client instance
    objects.
    """
    _transport_registry = OrderedDict()
    _transport_registry['rest'] = OperationsRestTransport

    def get_transport_class(cls, label: Optional[str]=None) -> Type[OperationsTransport]:
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