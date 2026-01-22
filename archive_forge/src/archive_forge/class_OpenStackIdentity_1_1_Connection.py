import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
class OpenStackIdentity_1_1_Connection(OpenStackIdentityConnection):
    """
    Connection class for Keystone API v1.1.
    """
    responseCls = OpenStackAuthResponse
    name = 'OpenStack Identity API v1.1'
    auth_version = '1.1'

    def authenticate(self, force=False):
        if not self._is_authentication_needed(force=force):
            return self
        reqbody = json.dumps({'credentials': {'username': self.user_id, 'key': self.key}})
        resp = self.request('/v1.1/auth', data=reqbody, headers={}, method='POST')
        if resp.status == httplib.UNAUTHORIZED:
            raise InvalidCredsError()
        elif resp.status != httplib.OK:
            body = 'code: {} body:{}'.format(resp.status, resp.body)
            raise MalformedResponseError('Malformed response', body=body, driver=self.driver)
        else:
            try:
                body = json.loads(resp.body)
            except Exception as e:
                raise MalformedResponseError('Failed to parse JSON', e)
            try:
                expires = body['auth']['token']['expires']
                self._cache_auth_context(OpenStackAuthenticationContext(body['auth']['token']['id'], expiration=parse_date(expires), urls=body['auth']['serviceCatalog']))
            except KeyError as e:
                raise MalformedResponseError('Auth JSON response is                                              missing required elements', e)
        return self