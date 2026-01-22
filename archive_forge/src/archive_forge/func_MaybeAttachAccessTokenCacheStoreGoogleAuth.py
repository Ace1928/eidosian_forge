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
def MaybeAttachAccessTokenCacheStoreGoogleAuth(credentials, access_token_file=None):
    """Attaches access token cache to given credentials if no store set.

  Note that credentials themselves will not be persisted only access token. Use
  this whenever access token caching is desired, yet credentials themselves
  should not be persisted.

  For external account and external account authorized user non-impersonated
  credentials, the provided credentials should have been instantiated with
  the client_id and client_secret in order to retrieve the account ID from the
  3PI token instrospection endpoint.

  Args:
    credentials: google.auth.credentials.Credentials.
    access_token_file: str, optional path to use for access token storage.

  Returns:
    google.auth.credentials.Credentials, reloaded credentials.
  """
    account_id = getattr(credentials, 'service_account_email', None)
    if not account_id and (isinstance(credentials, google_auth_external_account.Credentials) or isinstance(credentials, google_auth_external_account_authorized_user.Credentials)):
        account_id = c_introspect.GetExternalAccountId(credentials)
    elif not account_id:
        account_id = hashlib.sha256(six.ensure_binary(credentials.refresh_token)).hexdigest()
    access_token_cache = AccessTokenCache(access_token_file or config.Paths().access_token_db_path)
    store = AccessTokenStoreGoogleAuth(access_token_cache, account_id, credentials)
    credentials = store.Get()
    orig_refresh = credentials.refresh

    def _WrappedRefresh(request):
        orig_refresh(request)
        credentials.id_tokenb64 = getattr(credentials, '_id_token', None)
        store.Put()
    credentials.refresh = _WrappedRefresh
    return credentials