import hmac
import json
import base64
import datetime
from hashlib import sha256
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, ProviderError, InvalidCredsError
class AuroraDNSConnection(ConnectionUserAndKey):
    host = API_HOST
    responseCls = AuroraDNSResponse

    def calculate_auth_signature(self, secret_key, method, url, timestamp):
        b64_hmac = base64.b64encode(hmac.new(b(secret_key), b(method) + b(url) + b(timestamp), digestmod=sha256).digest())
        return b64_hmac.decode('utf-8')

    def gen_auth_header(self, api_key, secret_key, method, url, timestamp):
        signature = self.calculate_auth_signature(secret_key, method, url, timestamp)
        auth_b64 = base64.b64encode(b('{}:{}'.format(api_key, signature)))
        return 'AuroraDNSv1 %s' % auth_b64.decode('utf-8')

    def request(self, action, params=None, data='', headers=None, method='GET'):
        if not headers:
            headers = {}
        if not params:
            params = {}
        if method in ('POST', 'PUT'):
            headers = {'Content-Type': 'application/json; charset=UTF-8'}
        t = datetime.datetime.utcnow()
        timestamp = t.strftime('%Y%m%dT%H%M%SZ')
        headers['X-AuroraDNS-Date'] = timestamp
        headers['Authorization'] = self.gen_auth_header(self.user_id, self.key, method, action, timestamp)
        return super().request(action=action, params=params, data=data, method=method, headers=headers)