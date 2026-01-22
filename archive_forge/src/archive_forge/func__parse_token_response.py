import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def _parse_token_response(self, response, cache_it=False, raise_ambiguous_version_error=True):
    """
        Parse a response from /v3/auth/tokens.

        :param cache_it: Should we cache the authentication context?
        :type cache_it: ``bool``

        :param raise_ambiguous_version_error: Should an ambiguous version
            error be raised on a 300 response?
        :type raise_ambiguous_version_error: ``bool``
        """
    if response.status == httplib.UNAUTHORIZED:
        raise InvalidCredsError()
    elif response.status in [httplib.OK, httplib.CREATED]:
        headers = response.headers
        try:
            body = json.loads(response.body)
        except Exception as e:
            raise MalformedResponseError('Failed to parse JSON', e)
        try:
            roles = self._to_roles(body['token']['roles'])
        except Exception:
            roles = []
        try:
            expires = parse_date(body['token']['expires_at'])
            token = headers['x-subject-token']
            if cache_it:
                self._cache_auth_context(OpenStackAuthenticationContext(token, expiration=expires))
            self.auth_token = token
            self.auth_token_expires = expires
            self.urls = body['token'].get('catalog', None)
            self.auth_user_info = body['token'].get('user', None)
            self.auth_user_roles = roles
        except KeyError as e:
            raise MalformedResponseError('Auth JSON response is                                              missing required elements', e)
    elif raise_ambiguous_version_error and response.status == 300:
        raise LibcloudError('Auth request returned ambiguous version error, tryusing the version specific URL to connect, e.g. identity/v3/auth/tokens')
    else:
        body = 'code: {} body:{}'.format(response.status, response.body)
        raise MalformedResponseError('Malformed response', body=body, driver=self.driver)