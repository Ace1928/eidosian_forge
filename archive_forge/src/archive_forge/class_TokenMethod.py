from keystoneauth1.identity.v3 import base
class TokenMethod(base.AuthMethod):
    """Construct an Auth plugin to fetch a token from a token.

    :param string token: Token for authentication.
    """
    _method_parameters = ['token']

    def get_auth_data(self, session, auth, headers, **kwargs):
        headers['X-Auth-Token'] = self.token
        return ('token', {'id': self.token})

    def get_cache_id_elements(self):
        return {'token_token': self.token}