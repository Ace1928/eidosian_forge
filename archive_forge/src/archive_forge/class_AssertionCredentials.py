import collections
import copy
import datetime
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import six
from six.moves import http_client
from six.moves import urllib
import oauth2client
from oauth2client import _helpers
from oauth2client import _pkce
from oauth2client import clientsecrets
from oauth2client import transport
class AssertionCredentials(GoogleCredentials):
    """Abstract Credentials object used for OAuth 2.0 assertion grants.

    This credential does not require a flow to instantiate because it
    represents a two legged flow, and therefore has all of the required
    information to generate and refresh its own access tokens. It must
    be subclassed to generate the appropriate assertion string.

    AssertionCredentials objects may be safely pickled and unpickled.
    """

    @_helpers.positional(2)
    def __init__(self, assertion_type, user_agent=None, token_uri=oauth2client.GOOGLE_TOKEN_URI, revoke_uri=oauth2client.GOOGLE_REVOKE_URI, **unused_kwargs):
        """Constructor for AssertionFlowCredentials.

        Args:
            assertion_type: string, assertion type that will be declared to the
                            auth server
            user_agent: string, The HTTP User-Agent to provide for this
                        application.
            token_uri: string, URI for token endpoint. For convenience defaults
                       to Google's endpoints but any OAuth 2.0 provider can be
                       used.
            revoke_uri: string, URI for revoke endpoint.
        """
        super(AssertionCredentials, self).__init__(None, None, None, None, None, token_uri, user_agent, revoke_uri=revoke_uri)
        self.assertion_type = assertion_type

    def _generate_refresh_request_body(self):
        assertion = self._generate_assertion()
        body = urllib.parse.urlencode({'assertion': assertion, 'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer'})
        return body

    def _generate_assertion(self):
        """Generate assertion string to be used in the access token request."""
        raise NotImplementedError

    def _revoke(self, http):
        """Revokes the access_token and deletes the store if available.

        Args:
            http: an object to be used to make HTTP requests.
        """
        self._do_revoke(http, self.access_token)

    def sign_blob(self, blob):
        """Cryptographically sign a blob (of bytes).

        Args:
            blob: bytes, Message to be signed.

        Returns:
            tuple, A pair of the private key ID used to sign the blob and
            the signed contents.
        """
        raise NotImplementedError('This method is abstract.')