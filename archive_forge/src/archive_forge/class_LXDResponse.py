import os
import re
import base64
import collections
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey, KeyCertificateConnection
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import StorageVolume
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.container.providers import Provider
class LXDResponse(JsonResponse):
    valid_response_codes = [httplib.OK, httplib.ACCEPTED, httplib.CREATED, httplib.NO_CONTENT]

    def parse_body(self):
        if len(self.body) == 0 and (not self.parse_zero_length_body):
            return self.body
        try:
            content_type = self.headers.get('content-type', 'application/json')
            if content_type == 'application/json' or content_type == '':
                if self.headers.get('transfer-encoding') == 'chunked' and 'fromImage' in self.request.url:
                    body = [json.loads(chunk) for chunk in self.body.strip().replace('\r', '').split('\n')]
                else:
                    body = json.loads(self.body)
            else:
                body = self.body
        except ValueError:
            m = re.search('Error: (.+?)"', self.body)
            if m:
                error_msg = m.group(1)
                raise Exception(error_msg)
            else:
                msg = 'ConnectionError: Failed to parse JSON response (body=%s)' % self.body
                raise Exception(msg)
        return body

    def parse_error(self):
        if self.status == 401:
            raise InvalidCredsError('Invalid credentials')
        return self.body

    def success(self):
        return self.status in self.valid_response_codes