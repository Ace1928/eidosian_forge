import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
class OpenStackIdentity_1_0_Connection(OpenStackIdentityConnection):
    """
    Connection class for Keystone API v1.0.
    """
    responseCls = OpenStackAuthResponse
    name = 'OpenStack Identity API v1.0'
    auth_version = '1.0'

    def authenticate(self, force=False):
        if not self._is_authentication_needed(force=force):
            return self
        headers = {'X-Auth-User': self.user_id, 'X-Auth-Key': self.key}
        resp = self.request('/v1.0', headers=headers, method='GET')
        if resp.status == httplib.UNAUTHORIZED:
            raise InvalidCredsError()
        elif resp.status not in [httplib.NO_CONTENT, httplib.OK]:
            body = 'code: {} body:{} headers:{}'.format(resp.status, resp.body, resp.headers)
            raise MalformedResponseError('Malformed response', body=body, driver=self.driver)
        else:
            headers = resp.headers
            self.urls = {}
            self.urls['cloudServers'] = [{'publicURL': headers.get('x-server-management-url', None)}]
            self.urls['cloudFilesCDN'] = [{'publicURL': headers.get('x-cdn-management-url', None)}]
            self.urls['cloudFiles'] = [{'publicURL': headers.get('x-storage-url', None)}]
            self.auth_token = headers.get('x-auth-token', None)
            self.auth_user_info = None
            if not self.auth_token:
                raise MalformedResponseError('Missing X-Auth-Token in response headers')
        return self