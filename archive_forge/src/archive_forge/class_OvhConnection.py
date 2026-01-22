import time
import hashlib
from typing import List
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.utils.connection import get_response_object
class OvhConnection(ConnectionUserAndKey):
    """
    A connection to the Ovh API

    Wraps SSL connections to the Ovh API, automagically injecting the
    parameters that the API needs for each request.
    """
    host = API_HOST
    request_path = API_ROOT
    responseCls = OvhResponse
    timestamp = None
    ua = []
    LOCATIONS = LOCATIONS
    _timedelta = None
    allow_insecure = True

    def __init__(self, user_id, *args, **kwargs):
        region = kwargs.pop('region', '')
        if region:
            self.host = '{}.{}'.format(region, API_HOST).lstrip('.')
        else:
            self.host = API_HOST
        self.consumer_key = kwargs.pop('ex_consumer_key', None)
        if self.consumer_key is None:
            consumer_key_json = self.request_consumer_key(user_id)
            msg = 'Your consumer key isn\'t validated, go to \'%(validationUrl)s\' for valid it. After instantiate your driver with "ex_consumer_key=\'%(consumerKey)s\'".' % consumer_key_json
            raise OvhException(msg)
        super().__init__(user_id, *args, **kwargs)

    def request_consumer_key(self, user_id):
        action = self.request_path + '/auth/credential'
        data = json.dumps({'accessRules': DEFAULT_ACCESS_RULES, 'redirection': 'http://ovh.com'})
        headers = {'Content-Type': 'application/json', 'X-Ovh-Application': user_id}
        httpcon = LibcloudConnection(host=self.host, port=443)
        try:
            httpcon.request(method='POST', url=action, body=data, headers=headers)
        except Exception as e:
            handle_and_rethrow_user_friendly_invalid_region_error(host=self.host, e=e)
        response = OvhResponse(httpcon.getresponse(), httpcon)
        if response.status == httplib.UNAUTHORIZED:
            raise InvalidCredsError()
        json_response = response.parse_body()
        httpcon.close()
        return json_response

    def get_timestamp(self):
        if not self._timedelta:
            url = 'https://{}{}/auth/time'.format(self.host, API_ROOT)
            response = get_response_object(url=url, method='GET', headers={})
            if not response or not response.body:
                raise Exception('Failed to get current time from Ovh API')
            timestamp = int(response.body)
            self._timedelta = timestamp - int(time.time())
        return int(time.time()) + self._timedelta

    def make_signature(self, method, action, params, data, timestamp):
        full_url = 'https://{}{}'.format(self.host, action)
        if params:
            full_url += '?'
            for key, value in params.items():
                full_url += '{}={}&'.format(key, value)
            full_url = full_url[:-1]
        sha1 = hashlib.sha1()
        base_signature = '+'.join([self.key, self.consumer_key, method.upper(), full_url, data if data else '', str(timestamp)])
        sha1.update(base_signature.encode())
        signature = '$1$' + sha1.hexdigest()
        return signature

    def add_default_params(self, params):
        return params

    def add_default_headers(self, headers):
        headers.update({'X-Ovh-Application': self.user_id, 'X-Ovh-Consumer': self.consumer_key, 'Content-type': 'application/json'})
        return headers

    def request(self, action, params=None, data=None, headers=None, method='GET', raw=False):
        data = json.dumps(data) if data else None
        timestamp = self.get_timestamp()
        signature = self.make_signature(method, action, params, data, timestamp)
        headers = headers or {}
        headers.update({'X-Ovh-Timestamp': timestamp, 'X-Ovh-Signature': signature})
        try:
            return super().request(action, params=params, data=data, headers=headers, method=method, raw=raw)
        except Exception as e:
            handle_and_rethrow_user_friendly_invalid_region_error(host=self.host, e=e)