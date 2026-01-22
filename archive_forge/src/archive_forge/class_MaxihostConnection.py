from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import InvalidCredsError
class MaxihostConnection(ConnectionKey):
    """
    Connection class for the Maxihost driver.
    """
    host = 'api.maxihost.com'
    responseCls = MaxihostResponse

    def add_default_headers(self, headers):
        """
        Add headers that are necessary for every request

        This method adds apikey to the request.
        """
        headers['Authorization'] = 'Bearer %s' % self.key
        headers['Content-Type'] = 'application/json'
        headers['Accept'] = 'application/vnd.maxihost.v1.1+json'
        return headers