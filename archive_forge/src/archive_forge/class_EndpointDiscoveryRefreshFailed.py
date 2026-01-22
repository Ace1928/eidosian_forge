import logging
import time
import weakref
from botocore import xform_name
from botocore.exceptions import BotoCoreError, ConnectionError, HTTPClientError
from botocore.model import OperationNotFoundError
from botocore.utils import CachedProperty
class EndpointDiscoveryRefreshFailed(EndpointDiscoveryException):
    """Endpoint Discovery failed to the refresh the known endpoints."""
    fmt = 'Endpoint Discovery failed to refresh the required endpoints.'