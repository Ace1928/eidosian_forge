import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
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