import json
import re
import typing as ty
from requests import exceptions as _rex
class ServiceDiscoveryException(SDKException):
    """The service cannot be discovered."""