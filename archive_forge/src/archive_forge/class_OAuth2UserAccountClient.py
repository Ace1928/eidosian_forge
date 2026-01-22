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
class OAuth2UserAccountClient(OAuth2Client):
    """An OAuth2 client."""

    def __init__(self, token_uri, client_id, client_secret, refresh_token, auth_uri=None, access_token_cache=None, datetime_strategy=datetime.datetime, disable_ssl_certificate_validation=False, proxy_host=None, proxy_port=None, proxy_user=None, proxy_pass=None, ca_certs_file=None):
        """Creates an OAuth2UserAccountClient.

    Args:
      token_uri: The URI used to refresh access tokens.
      client_id: The OAuth2 client ID of this client.
      client_secret: The OAuth2 client secret of this client.
      refresh_token: The token used to refresh the access token.
      auth_uri: The URI for OAuth2 authorization.
      access_token_cache: An optional instance of a TokenCache. If omitted or
          None, an InMemoryTokenCache is used.
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
        super(OAuth2UserAccountClient, self).__init__(cache_key_base=refresh_token, auth_uri=auth_uri, token_uri=token_uri, access_token_cache=access_token_cache, datetime_strategy=datetime_strategy, disable_ssl_certificate_validation=disable_ssl_certificate_validation, proxy_host=proxy_host, proxy_port=proxy_port, proxy_user=proxy_user, proxy_pass=proxy_pass, ca_certs_file=ca_certs_file)
        self.token_uri = token_uri
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token

    def GetCredentials(self):
        """Fetches a credentials objects from the provider's token endpoint."""
        access_token = self.GetAccessToken()
        credentials = reauth_creds.Oauth2WithReauthCredentials(access_token.token, self.client_id, self.client_secret, self.refresh_token, access_token.expiry, self.token_uri, None)
        return credentials

    @retry_decorator.retry(GsAccessTokenRefreshError, tries=boto.config.get('OAuth2', 'oauth2_refresh_retries', 6), timeout_secs=1)
    def FetchAccessToken(self, rapt_token=None):
        """Fetches an access token from the provider's token endpoint.

    Fetches an access token from this client's OAuth2 provider's token endpoint.

    Args:
      rapt_token: (str) The RAPT to be passed when refreshing the access token.

    Returns:
      The fetched AccessToken.
    """
        try:
            http = self.CreateHttpRequest()
            credentials = reauth_creds.Oauth2WithReauthCredentials(None, self.client_id, self.client_secret, self.refresh_token, None, self.token_uri, None, scopes=RAPT_SCOPES, rapt_token=rapt_token)
            credentials.refresh(http)
            return AccessToken(credentials.access_token, credentials.token_expiry, datetime_strategy=self.datetime_strategy, rapt_token=credentials.rapt_token)
        except oauth2client.client.AccessTokenRefreshError as e:
            if 'Invalid response 403' in e.message:
                raise GsAccessTokenRefreshError(e)
            elif 'invalid_grant' in e.message:
                LOG.info('\nAttempted to retrieve an access token from an invalid refresh token. Two common\ncases in which you will see this error are:\n1. Your refresh token was revoked.\n2. Your refresh token was typed incorrectly.\n')
                raise GsInvalidRefreshTokenError(e)
            else:
                raise