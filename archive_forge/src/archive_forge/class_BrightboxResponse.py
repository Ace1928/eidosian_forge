from libcloud.utils.py3 import b, httplib, base64_encode_string
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.compute.types import InvalidCredsError
class BrightboxResponse(JsonResponse):

    def success(self):
        return httplib.OK <= self.status < httplib.BAD_REQUEST

    def parse_body(self):
        if self.headers['content-type'].split(';')[0] == 'application/json':
            return super().parse_body()
        else:
            return self.body

    def parse_error(self):
        response = super().parse_body()
        if 'error' in response:
            if response['error'] in ['invalid_client', 'unauthorized_client']:
                raise InvalidCredsError(response['error'])
            return response['error']
        elif 'error_name' in response:
            return '{}: {}'.format(response['error_name'], response['errors'][0])
        return self.body