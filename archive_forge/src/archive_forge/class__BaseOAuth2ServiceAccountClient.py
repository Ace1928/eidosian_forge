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
class _BaseOAuth2ServiceAccountClient(OAuth2Client):
    """Base class for OAuth2ServiceAccountClients.

  Args:
    client_id: The OAuth2 client ID of this client.
    access_token_cache: An optional instance of a TokenCache. If omitted or
        None, an InMemoryTokenCache is used.
    auth_uri: The URI for OAuth2 authorization.
    token_uri: The URI used to refresh access tokens.
    datetime_strategy: datetime module strategy to use.
    disable_ssl_certificate_validation: True if certifications should not be
        validated.
    proxy_host: An optional string specifying the host name of an HTTP proxy
        to be used.
    proxy_port: An optional int specifying the port number of an HTTP proxy
        to be used.
    proxy_user: An optional string specifying the user name for interacting
        with the HTTP proxy.
    proxy_pass: An optional string specifying the password for interacting
        with the HTTP proxy.
    ca_certs_file: The cacerts.txt file to use.
  """

    def __init__(self, client_id, access_token_cache=None, auth_uri=None, token_uri=None, datetime_strategy=datetime.datetime, disable_ssl_certificate_validation=False, proxy_host=None, proxy_port=None, proxy_user=None, proxy_pass=None, ca_certs_file=None):
        super(_BaseOAuth2ServiceAccountClient, self).__init__(cache_key_base=client_id, auth_uri=auth_uri, token_uri=token_uri, access_token_cache=access_token_cache, datetime_strategy=datetime_strategy, disable_ssl_certificate_validation=disable_ssl_certificate_validation, proxy_host=proxy_host, proxy_port=proxy_port, proxy_user=proxy_user, proxy_pass=proxy_pass, ca_certs_file=ca_certs_file)
        self._client_id = client_id

    def FetchAccessToken(self, rapt_token=None):
        credentials = self.GetCredentials()
        http = self.CreateHttpRequest()
        credentials.refresh(http)
        return AccessToken(credentials.access_token, credentials.token_expiry, datetime_strategy=self.datetime_strategy, rapt_token=rapt_token)