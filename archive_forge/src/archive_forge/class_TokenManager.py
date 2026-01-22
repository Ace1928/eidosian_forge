from keystoneauth1 import exceptions
from keystoneauth1 import plugin
from keystoneclient import access
from keystoneclient import base
from keystoneclient.i18n import _
class TokenManager(base.Manager):
    resource_class = Token

    def authenticate(self, username=None, tenant_id=None, tenant_name=None, password=None, token=None, return_raw=False):
        if token:
            params = {'auth': {'token': {'id': token}}}
        elif username and password:
            params = {'auth': {'passwordCredentials': {'username': username, 'password': password}}}
        else:
            raise ValueError(_('A username and password or token is required.'))
        if tenant_id:
            params['auth']['tenantId'] = tenant_id
        elif tenant_name:
            params['auth']['tenantName'] = tenant_name
        args = ['/tokens', params, 'access']
        kwargs = {'return_raw': return_raw, 'log': False}
        try:
            token_ref = self._post(*args, **kwargs)
        except exceptions.EndpointNotFound:
            kwargs['endpoint_filter'] = {'interface': plugin.AUTH_INTERFACE}
            token_ref = self._post(*args, **kwargs)
        return token_ref

    def delete(self, token):
        return self._delete('/tokens/%s' % base.getid(token))

    def endpoints(self, token):
        return self._get('/tokens/%s/endpoints' % base.getid(token), 'token')

    def validate(self, token):
        """Validate a token.

        :param token: Token to be validated.

        :rtype: :py:class:`.Token`

        """
        return self._get('/tokens/%s' % base.getid(token), 'access')

    def get_token_data(self, token):
        """Fetch the data about a token from the identity server.

        :param str token: The token id.

        :rtype: dict
        """
        url = '/tokens/%s' % token
        resp, body = self.client.get(url)
        return body

    def validate_access_info(self, token):
        """Validate a token.

        :param token: Token to be validated. This can be an instance of
                      :py:class:`keystoneclient.access.AccessInfo` or a string
                      token_id.

        :rtype: :py:class:`keystoneclient.access.AccessInfoV2`

        """

        def calc_id(token):
            if isinstance(token, access.AccessInfo):
                return token.auth_token
            return base.getid(token)
        token_id = calc_id(token)
        body = self.get_token_data(token_id)
        return access.AccessInfo.factory(auth_token=token_id, body=body)

    def get_revoked(self):
        """Return the revoked tokens response.

        The response will be a dict containing 'signed' which is a CMS-encoded
        document.

        """
        resp, body = self.client.get('/tokens/revoked')
        return body