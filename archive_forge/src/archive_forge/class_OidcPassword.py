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
class OidcPassword(_OidcBase):
    """Implementation for OpenID Connect Resource Owner Password Credential."""
    grant_type = 'password'

    def __init__(self, auth_url, identity_provider, protocol, client_id, client_secret, access_token_endpoint=None, discovery_endpoint=None, access_token_type='access_token', username=None, password=None, **kwargs):
        """The OpenID Password plugin expects the following.

        :param username: Username used to authenticate
        :type username: string

        :param password: Password used to authenticate
        :type password: string
        """
        super(OidcPassword, self).__init__(auth_url=auth_url, identity_provider=identity_provider, protocol=protocol, client_id=client_id, client_secret=client_secret, access_token_endpoint=access_token_endpoint, discovery_endpoint=discovery_endpoint, access_token_type=access_token_type, **kwargs)
        self.username = username
        self.password = password

    def get_payload(self, session):
        """Get an authorization grant for the "password" grant type.

        :param session: a session object to send out HTTP requests.
        :type session: keystoneauth1.session.Session

        :returns: a python dictionary containing the payload to be exchanged
        :rtype: dict
        """
        payload = {'username': self.username, 'password': self.password, 'scope': self.scope, 'client_id': self.client_id}
        return payload