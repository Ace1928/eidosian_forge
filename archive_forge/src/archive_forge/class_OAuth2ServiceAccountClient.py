from __future__ import absolute_import
import datetime
import errno
from hashlib import sha1
import json
import logging
import os
import socket
import tempfile
import threading
import boto
import httplib2
import oauth2client.client
import oauth2client.service_account
from google_reauth import reauth_creds
import retry_decorator.retry_decorator
import six
from six import BytesIO
from six.moves import urllib
class OAuth2ServiceAccountClient(_BaseOAuth2ServiceAccountClient):
    """An OAuth2 service account client using .p12 or .pem keys."""

    def __init__(self, client_id, private_key, password, access_token_cache=None, auth_uri=None, token_uri=None, datetime_strategy=datetime.datetime, disable_ssl_certificate_validation=False, proxy_host=None, proxy_port=None, proxy_user=None, proxy_pass=None, ca_certs_file=None):
        """Creates an OAuth2ServiceAccountClient.

    Args:
      client_id: The OAuth2 client ID of this client.
      private_key: The private key associated with this service account.
      password: The private key password used for the crypto signer.

    Keyword arguments match the _BaseOAuth2ServiceAccountClient class.
    """
        super(OAuth2ServiceAccountClient, self).__init__(client_id, auth_uri=auth_uri, token_uri=token_uri, access_token_cache=access_token_cache, datetime_strategy=datetime_strategy, disable_ssl_certificate_validation=disable_ssl_certificate_validation, proxy_host=proxy_host, proxy_port=proxy_port, proxy_user=proxy_user, proxy_pass=proxy_pass, ca_certs_file=ca_certs_file)
        self._private_key = private_key
        self._password = password

    def GetCredentials(self):
        if oauth2client.client.HAS_CRYPTO:
            return _ServiceAccountCredentials.from_p12_keyfile_buffer(self._client_id, BytesIO(self._private_key), private_key_password=self._password, scopes=DEFAULT_SCOPE, token_uri=self.token_uri)
        else:
            raise MissingDependencyError('Service account authentication requires PyOpenSSL. Please install this library and try again.')