from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.gandi import BaseObject
from libcloud.common.types import LibcloudError, InvalidCredsError
class LinodeConnectionV4(ConnectionKey):
    """
    A connection to the Linode API

    Wraps SSL connections to the Linode API
    """
    host = API_HOST
    responseCls = LinodeResponseV4

    def add_default_headers(self, headers):
        """
        Add headers that are necessary for every request

        This method adds ``token`` to the request.
        """
        headers['Authorization'] = 'Bearer %s' % self.key
        headers['Content-Type'] = 'application/json'
        return headers

    def add_default_params(self, params):
        """
        Add parameters that are necessary for every request

        This method adds ``page_size`` to the request to reduce the total
        number of paginated requests to the API.
        """
        params['page_size'] = 25
        return params