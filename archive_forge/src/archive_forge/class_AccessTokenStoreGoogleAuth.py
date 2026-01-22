from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import base64
import collections
import copy
import datetime
import enum
import hashlib
import json
import os
import sqlite3
from google.auth import compute_engine as google_auth_compute_engine
from google.auth import credentials as google_auth_creds
from google.auth import exceptions as google_auth_exceptions
from google.auth import external_account as google_auth_external_account
from google.auth import external_account_authorized_user as google_auth_external_account_authorized_user
from google.auth import impersonated_credentials as google_auth_impersonated
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import devshell as c_devshell
from googlecloudsdk.core.credentials import exceptions as c_exceptions
from googlecloudsdk.core.credentials import introspect as c_introspect
from googlecloudsdk.core.util import files
from oauth2client import client
from oauth2client import service_account
from oauth2client.contrib import gce as oauth2client_gce
import six
class AccessTokenStoreGoogleAuth(object):
    """google-auth adapted for access token cache.

  This class works with google-auth credentials and serializes its short lived
  tokens, including access token, token expiry, rapt token, id token into the
  access token cache.
  """

    def __init__(self, access_token_cache, account_id, credentials):
        """Sets up token store for given account.

    Args:
      access_token_cache: AccessTokenCache, cache for access tokens.
      account_id: str, account for which token is stored.
      credentials: google.auth.credentials.Credentials, credentials of account
        of account_id.
    """
        self._access_token_cache = access_token_cache
        self._formatted_account_id = _AccountIdFormatter.GetFormattedAccountId(account_id, credentials)
        self._credentials = credentials

    def Get(self):
        """Gets credentials with short lived tokens from the internal cache.

    Retrieves the short lived tokens from the internal access token cache,
    populates the credentials with these tokens and returns the credentials.

    Returns:
       google.auth.credentials.Credentials
    """
        token_data = self._access_token_cache.Load(self._formatted_account_id)
        if token_data:
            access_token, token_expiry, rapt_token, id_token = token_data
            if UseSelfSignedJwt(self._credentials):
                self._credentials.token = None
                self._credentials.expiry = None
                self._credentials._rapt_token = None
            else:
                self._credentials.token = access_token
                self._credentials.expiry = token_expiry
                self._credentials._rapt_token = rapt_token
            self._credentials._id_token = id_token
            self._credentials.id_tokenb64 = id_token
        return self._credentials

    def Put(self):
        """Puts the short lived tokens of the credentials to the internal cache."""
        id_token = getattr(self._credentials, 'id_tokenb64', None) or getattr(self._credentials, '_id_token', None)
        expiry = getattr(self._credentials, 'expiry', None)
        rapt_token = getattr(self._credentials, 'rapt_token', None)
        access_token = getattr(self._credentials, 'token', None)
        if UseSelfSignedJwt(self._credentials):
            access_token = None
            expiry = None
            rapt_token = None
            token_data = self._access_token_cache.Load(self._formatted_account_id)
            if token_data:
                access_token, expiry, rapt_token, _ = token_data
        self._access_token_cache.Store(self._formatted_account_id, access_token, expiry, rapt_token, id_token)

    def Delete(self):
        """Removes the tokens of the account from the internal cache."""
        self._access_token_cache.Remove(self._formatted_account_id)