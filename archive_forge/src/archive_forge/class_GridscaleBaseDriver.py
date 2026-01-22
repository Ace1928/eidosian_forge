from libcloud.utils.py3 import httplib
from libcloud.common.base import BaseDriver, JsonResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
class GridscaleBaseDriver(BaseDriver):
    name = 'gridscale'
    website = 'https://gridscale.io'
    connectionCls = GridscaleConnection

    def __init__(self, user_id, key, **kwargs):
        super().__init__(user_id, key, **kwargs)

    def _sync_request(self, data=None, endpoint=None, method='GET'):
        raw_result = self.connection.request(endpoint, data=data, method=method)
        return raw_result