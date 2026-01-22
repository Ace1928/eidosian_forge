from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError
def add_default_headers(self, headers):
    if self.shopper_id is not None:
        headers['X-Shopper-Id'] = self.shopper_id
    headers['Content-type'] = 'application/json'
    headers['Authorization'] = 'sso-key {}:{}'.format(self.key, self.secret)
    return headers