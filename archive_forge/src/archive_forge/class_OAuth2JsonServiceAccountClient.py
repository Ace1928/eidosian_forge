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
class OAuth2JsonServiceAccountClient(_BaseOAuth2ServiceAccountClient):
    """An OAuth2 service account client using .json keys."""

    def __init__(self, json_key_dict, access_token_cache=None, auth_uri=None, token_uri=None, datetime_strategy=datetime.datetime, disable_ssl_certificate_validation=False, proxy_host=None, proxy_port=None, proxy_user=None, proxy_pass=None, ca_certs_file=None):
        """Creates an OAuth2JsonServiceAccountClient.

    Args:
      json_key_dict: dictionary from the json private key file. Includes:
          client_id: The OAuth2 client ID of this client.
          client_email: The email associated with this client.
          private_key_id: The private key id associated with this service
              account.
          private_key_pkcs8_text: The pkcs8 text containing the private key
              data.

    Keyword arguments match the _BaseOAuth2ServiceAccountClient class.
    """
        super(OAuth2JsonServiceAccountClient, self).__init__(json_key_dict['client_id'], auth_uri=auth_uri, token_uri=token_uri, access_token_cache=access_token_cache, datetime_strategy=datetime_strategy, disable_ssl_certificate_validation=disable_ssl_certificate_validation, proxy_host=proxy_host, proxy_port=proxy_port, proxy_user=proxy_user, proxy_pass=proxy_pass, ca_certs_file=ca_certs_file)
        self._json_key_dict = json_key_dict
        self._service_account_email = json_key_dict['client_email']
        self._private_key_id = json_key_dict['private_key_id']
        self._private_key_pkcs8_text = json_key_dict['private_key']

    def GetCredentials(self):
        return _ServiceAccountCredentials.from_json_keyfile_dict(self._json_key_dict, scopes=[DEFAULT_SCOPE], token_uri=self.token_uri)