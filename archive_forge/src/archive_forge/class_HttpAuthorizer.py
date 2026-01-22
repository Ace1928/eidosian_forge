import base64
class HttpAuthorizer:
    """Handles authentication for HTTP requests.

    There are two ways to authenticate.

    The authorize_session() method is called once when the client is
    initialized. This works for authentication methods like Basic
    Auth.  The authorize_request is called for every HTTP request,
    which is useful for authentication methods like Digest and OAuth.

    The base class is a null authorizer which does not perform any
    authentication at all.
    """

    def authorizeSession(self, client):
        """Set up credentials for the entire session."""
        pass

    def authorizeRequest(self, absolute_uri, method, body, headers):
        """Set up credentials for a single request.

        This probably involves setting the Authentication header.
        """
        pass

    @property
    def user_agent_params(self):
        """Any parameters necessary to identify this user agent.

        By default this is an empty dict (because authentication
        details don't contain any information about the application
        making the request), but when a resource is protected by
        OAuth, the OAuth consumer name is part of the user agent.
        """
        return {}