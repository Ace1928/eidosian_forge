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
class AccessTokenCredentials(OAuth2Credentials):
    """Credentials object for OAuth 2.0.

    Credentials can be applied to an httplib2.Http object using the
    authorize() method, which then signs each request from that object
    with the OAuth 2.0 access token. This set of credentials is for the
    use case where you have acquired an OAuth 2.0 access_token from
    another place such as a JavaScript client or another web
    application, and wish to use it from Python. Because only the
    access_token is present it can not be refreshed and will in time
    expire.

    AccessTokenCredentials objects may be safely pickled and unpickled.

    Usage::

        credentials = AccessTokenCredentials('<an access token>',
            'my-user-agent/1.0')
        http = httplib2.Http()
        http = credentials.authorize(http)

    Raises:
        AccessTokenCredentialsExpired: raised when the access_token expires or
                                       is revoked.
    """

    def __init__(self, access_token, user_agent, revoke_uri=None):
        """Create an instance of OAuth2Credentials

        This is one of the few types if Credentials that you should contrust,
        Credentials objects are usually instantiated by a Flow.

        Args:
            access_token: string, access token.
            user_agent: string, The HTTP User-Agent to provide for this
                        application.
            revoke_uri: string, URI for revoke endpoint. Defaults to None; a
                        token can't be revoked if this is None.
        """
        super(AccessTokenCredentials, self).__init__(access_token, None, None, None, None, None, user_agent, revoke_uri=revoke_uri)

    @classmethod
    def from_json(cls, json_data):
        data = json.loads(_helpers._from_bytes(json_data))
        retval = AccessTokenCredentials(data['access_token'], data['user_agent'])
        return retval

    def _refresh(self, http):
        """Refreshes the access token.

        Args:
            http: unused HTTP object.

        Raises:
            AccessTokenCredentialsError: always
        """
        raise AccessTokenCredentialsError("The access_token is expired or invalid and can't be refreshed.")

    def _revoke(self, http):
        """Revokes the access_token and deletes the store if available.

        Args:
            http: an object to be used to make HTTP requests.
        """
        self._do_revoke(http, self.access_token)