from troveclient.compat import exceptions
class KeyStoneV3Authenticator(Authenticator):

    def __init__(self, client, type, url, username, password, tenant, region=None, service_type=None, service_name=None, service_url=None):
        super(KeyStoneV3Authenticator, self).__init__(client, type, url, username, password, tenant, region=region, service_type=service_type, service_name=service_name, service_url=service_url)

        class Auth(object):

            def __init__(self, auth_url, username, password, project_name):
                token_str = '/auth/tokens'
                if auth_url.endswith(token_str):
                    auth_url = auth_url[:-len(token_str)]
                self.auth_url = auth_url
                self._username = username
                self._password = password
                self._project_name = project_name
        self.auth = Auth(url, username, password, tenant)

    def authenticate(self):
        if self.url is None:
            raise exceptions.AuthUrlNotGiven()
        return self._v3_auth(self.url)

    def _v3_auth(self, url):
        """Authenticate against a v3.0 auth service."""
        body = {'auth': {'identity': {'methods': ['password'], 'password': {'user': {'domain': {'name': 'Default'}, 'name': self.username, 'password': self.password}}}}}
        if self.tenant:
            body['auth']['scope'] = {'project': {'domain': {'name': 'Default'}, 'name': self.tenant}}
        return self._authenticate(url, body)