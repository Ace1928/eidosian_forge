import json
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import ProviderError
class GandiLiveBaseError(ProviderError):
    """
    Exception class for Gandi Live driver
    """
    pass