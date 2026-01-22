from typing import Dict, List
from libcloud.common.base import JsonResponse, ConnectionKey
class NsOneConnection(ConnectionKey):
    host = API_HOST
    responseCls = NsOneResponse

    def add_default_headers(self, headers):
        headers['Content-Type'] = 'application/json'
        headers['X-NSONE-KEY'] = self.key
        return headers