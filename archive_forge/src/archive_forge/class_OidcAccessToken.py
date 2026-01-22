import abc
import base64
import hashlib
import os
import time
from urllib import parse as urlparse
import warnings
from keystoneauth1 import _utils as utils
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import federation
class OidcAccessToken(_OidcBase):
    """Implementation for OpenID Connect access token reuse."""

    def __init__(self, auth_url, identity_provider, protocol, access_token, **kwargs):
        """The OpenID Connect plugin based on the Access Token.

        It expects the following:

        :param auth_url: URL of the Identity Service
        :type auth_url: string

        :param identity_provider: Name of the Identity Provider the client
                                  will authenticate against
        :type identity_provider: string

        :param protocol: Protocol name as configured in keystone
        :type protocol: string

        :param access_token: OpenID Connect Access token
        :type access_token: string
        """
        super(OidcAccessToken, self).__init__(auth_url, identity_provider, protocol, client_id=None, client_secret=None, access_token_endpoint=None, access_token_type=None, **kwargs)
        self.access_token = access_token

    def get_payload(self, session):
        """OidcAccessToken does not require a payload."""
        return {}

    def get_unscoped_auth_ref(self, session):
        """Authenticate with OpenID Connect and get back claims.

        We exchange the access token upon accessing the protected Keystone
        endpoint (federated auth URL). This will trigger the OpenID Connect
        Provider to perform a user introspection and retrieve information
        (specified in the scope) about the user in the form of an OpenID
        Connect Claim. These claims will be sent to Keystone in the form of
        environment variables.

        :param session: a session object to send out HTTP requests.
        :type session: keystoneauth1.session.Session

        :returns: a token data representation
        :rtype: :py:class:`keystoneauth1.access.AccessInfoV3`
        """
        response = self._get_keystone_token(session, self.access_token)
        return access.create(resp=response)