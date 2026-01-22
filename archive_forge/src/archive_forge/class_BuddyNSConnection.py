from typing import Dict, List
from libcloud.common.base import JsonResponse, ConnectionKey
class BuddyNSConnection(ConnectionKey):
    host = API_HOST
    responseCls = BuddyNSResponse

    def add_default_headers(self, headers):
        headers['content-type'] = 'application/json'
        headers['Authorization'] = 'Token' + ' ' + self.key
        return headers