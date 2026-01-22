import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
class OpenStackAuthResponse(Response):

    def success(self):
        return self.status in [httplib.OK, httplib.CREATED, httplib.ACCEPTED, httplib.NO_CONTENT, httplib.MULTIPLE_CHOICES, httplib.UNAUTHORIZED, httplib.INTERNAL_SERVER_ERROR]

    def parse_body(self):
        if not self.body:
            return None
        if 'content-type' in self.headers:
            key = 'content-type'
        elif 'Content-Type' in self.headers:
            key = 'Content-Type'
        else:
            raise LibcloudError('Missing content-type header', driver=OpenStackIdentityConnection)
        content_type = self.headers[key]
        if content_type.find(';') != -1:
            content_type = content_type.split(';')[0]
        if content_type == 'application/json':
            try:
                data = json.loads(self.body)
            except Exception:
                driver = OpenStackIdentityConnection
                raise MalformedResponseError('Failed to parse JSON', body=self.body, driver=driver)
        elif content_type == 'text/plain':
            data = self.body
        else:
            data = self.body
        return data