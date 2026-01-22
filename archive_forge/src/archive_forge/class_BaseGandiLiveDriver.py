import json
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import ProviderError
class BaseGandiLiveDriver:
    """
    Gandi Live base driver
    """
    connectionCls = GandiLiveConnection
    name = 'GandiLive'