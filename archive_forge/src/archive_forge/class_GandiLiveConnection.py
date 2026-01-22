import json
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import ProviderError
class GandiLiveConnection(ConnectionKey):
    """
    Connection class for the Gandi Live driver
    """
    responseCls = GandiLiveResponse
    host = API_HOST

    def add_default_headers(self, headers):
        """
        Returns default headers as a dictionary.
        """
        headers['Content-Type'] = 'application/json'
        headers['X-Api-Key'] = self.key
        return headers

    def encode_data(self, data):
        """Encode data to JSON"""
        return json.dumps(data)