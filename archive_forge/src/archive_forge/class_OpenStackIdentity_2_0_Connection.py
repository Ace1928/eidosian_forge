import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
class OpenStackIdentity_2_0_Connection(OpenStackIdentityConnection):
    """
    Connection class for Keystone API v2.0.
    """
    responseCls = OpenStackAuthResponse
    name = 'OpenStack Identity API v1.0'
    auth_version = '2.0'

    def authenticate(self, auth_type='api_key', force=False):
        if not self._is_authentication_needed(force=force):
            return self
        if auth_type == 'api_key':
            return self._authenticate_2_0_with_api_key()
        elif auth_type == 'password':
            return self._authenticate_2_0_with_password()
        else:
            raise ValueError('Invalid value for auth_type argument')

    def _authenticate_2_0_with_api_key(self):
        data = {'auth': {'RAX-KSKEY:apiKeyCredentials': {'username': self.user_id, 'apiKey': self.key}}}
        if self.tenant_name:
            data['auth']['tenantName'] = self.tenant_name
        reqbody = json.dumps(data)
        return self._authenticate_2_0_with_body(reqbody)

    def _authenticate_2_0_with_password(self):
        data = {'auth': {'passwordCredentials': {'username': self.user_id, 'password': self.key}}}
        if self.tenant_name:
            data['auth']['tenantName'] = self.tenant_name
        reqbody = json.dumps(data)
        return self._authenticate_2_0_with_body(reqbody)

    def _authenticate_2_0_with_body(self, reqbody):
        resp = self.request('/v2.0/tokens', data=reqbody, headers={'Content-Type': 'application/json'}, method='POST')
        if resp.status == httplib.UNAUTHORIZED:
            raise InvalidCredsError()
        elif resp.status not in [httplib.OK, httplib.NON_AUTHORITATIVE_INFORMATION]:
            body = 'code: {} body: {}'.format(resp.status, resp.body)
            raise MalformedResponseError('Malformed response', body=body, driver=self.driver)
        else:
            body = resp.object
            try:
                access = body['access']
                expires = access['token']['expires']
                self._cache_auth_context(OpenStackAuthenticationContext(access['token']['id'], expiration=parse_date(expires), urls=access['serviceCatalog'], user=access.get('user', {})))
            except KeyError as e:
                raise MalformedResponseError('Auth JSON response is                                              missing required elements', e)
        return self

    def list_projects(self):
        response = self.authenticated_request('/v2.0/tenants', method='GET')
        result = self._to_projects(data=response.object['tenants'])
        return result

    def list_tenants(self):
        return self.list_projects()