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
class CredentialStoreWithCache(CredentialStore):
    """Implements CredentialStore for caching credentials information.

  Static credentials information, such as client ID and service account email,
  are stored in credentials.db. The short lived credentials tokens, such as
  access token, are cached in access_tokens.db.
  """

    def __init__(self, credential_store, access_token_cache):
        """Sets up credentials store for caching credentials.

    Args:
      credential_store: SqliteCredentialStore, for caching static credentials
        information, such as client ID, service account email, etc.
      access_token_cache: AccessTokenCache, for caching short lived credentials
        tokens, such as access token.
    """
        self._credential_store = credential_store
        self._access_token_cache = access_token_cache

    def _WrapCredentialsRefreshWithAutoCaching(self, credentials, store):
        """Wraps the refresh method of credentials with auto caching logic.

    For auto caching short lived tokens of google-auth credentials, such as
    access token, on credentials refresh.

    Args:
      credentials: google.auth.credentials.Credentials, the credentials updated
        by this method.
      store: AccessTokenStoreGoogleAuth, the store that caches the tokens of the
        input credentials.

    Returns:
      google.auth.credentials.Credentials, the updated credentials.
    """
        orig_refresh = credentials.refresh

        def _WrappedRefresh(request):
            orig_refresh(request)
            store.Put()
        credentials.refresh = _WrappedRefresh
        return credentials

    def GetAccounts(self):
        """Returns all the accounts stored in the cache."""
        return self._credential_store.GetAccounts()

    def Load(self, account_id, use_google_auth=True):
        """Loads the credentials of account_id from the cache.

    Args:
      account_id: string, ID of the account to load.
      use_google_auth: bool, True to load google-auth credentials if the type of
        the credentials is supported by the cache. False to load oauth2client
        credentials.

    Returns:
      1. None, if credentials are not found in the cache.
      2. google.auth.credentials.Credentials, if use_google_auth is true.
      3. client.OAuth2Credentials.
    """
        credentials = self._credential_store.Load(account_id, use_google_auth)
        if credentials is None:
            return None
        if IsOauth2ClientCredentials(credentials):
            store = AccessTokenStore(self._access_token_cache, account_id, credentials)
            credentials.set_store(store)
            return store.get()
        else:
            store = AccessTokenStoreGoogleAuth(self._access_token_cache, account_id, credentials)
            credentials = store.Get()
            return self._WrapCredentialsRefreshWithAutoCaching(credentials, store)

    def Store(self, account_id, credentials):
        """Stores credentials into the cache with account of account_id.

    Args:
      account_id: string, the account that will be associated with credentials
        in the cache.
      credentials: google.auth.credentials.Credentials or
        client.OAuth2Credentials, the credentials to be stored.
    """
        if IsOauth2ClientCredentials(credentials):
            store = AccessTokenStore(self._access_token_cache, account_id, credentials)
            credentials.set_store(store)
            store.put(credentials)
        else:
            store = AccessTokenStoreGoogleAuth(self._access_token_cache, account_id, credentials)
            store.Put()
        self._credential_store.Store(account_id, credentials)

    def Remove(self, account_id):
        """Removes credentials of account_id from the cache."""
        self._credential_store.Remove(account_id)
        self._access_token_cache.Remove(_AccountIdFormatter.GetFormattedAccountId(account_id))