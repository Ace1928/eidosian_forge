import base64
class BasicHttpAuthorizer(HttpAuthorizer):
    """Handles authentication for services that use HTTP Basic Auth."""

    def __init__(self, username, password):
        """Constructor.

        :param username: User to send as authorization for all requests.
        :param password: Password to send as authorization for all requests.
        """
        self.username = username
        self.password = password

    def authorizeRequest(self, absolute_uri, method, body, headers):
        """Set up credentials for a single request.

        This sets the authorization header with the username/password.
        """
        headers['authorization'] = 'Basic ' + base64.b64encode('%s:%s' % (self.username, self.password)).strip()

    def authorizeSession(self, client):
        client.add_credentials(self.username, self.password)