import time
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import urlparse, urlencode, basestring
from libcloud.common.base import BaseDriver, RawResponse, JsonResponse, ConnectionUserAndKey
class AzureJsonResponse(JsonResponse):

    def parse_error(self):
        b = self.parse_body()
        if isinstance(b, basestring):
            return b
        elif isinstance(b, dict) and 'error' in b:
            return '[{}] {}'.format(b['error'].get('code'), b['error'].get('message'))
        else:
            return str(b)