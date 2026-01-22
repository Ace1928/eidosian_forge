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
class AccessTokenCache(object):
    """Sqlite implementation of for access token cache.

  AccessTokenCache uses formatted_account_id instead of account_id in its APIs.
  The reason is that AccessTokenCache is used by AccessTokenStoreGoogleAuth,
  which is tied to a specific credential object. Either we let
  AccessTokenStoreGoogleAuth pass the credential's universe_domain to
  AccessTokenCache, or pass the formatted account id (which contains
  universe_domain). The latter is better since it is backward compatible and
  there is no need to introduce a new universe_domain parameter to all
  AccessTokenCache Load/Store/Remove APIs.
  See go/gcloud-multi-universe-auth-cache section 3.2, 3.3 for more details.
  """

    def __init__(self, store_file):
        self._cursor = _SqlCursor(store_file)
        self._Execute('CREATE TABLE IF NOT EXISTS "{}" (account_id TEXT PRIMARY KEY, access_token TEXT, token_expiry TIMESTAMP, rapt_token TEXT, id_token TEXT)'.format(_ACCESS_TOKEN_TABLE))
        try:
            self._Execute('SELECT id_token FROM "{}" LIMIT 1'.format(_ACCESS_TOKEN_TABLE))
        except sqlite3.OperationalError:
            self._Execute('ALTER TABLE "{}" ADD COLUMN id_token TEXT'.format(_ACCESS_TOKEN_TABLE))

    def _Execute(self, *args):
        with self._cursor as cur:
            cur.Execute(*args)

    def Load(self, formatted_account_id):
        """Load the tokens from the access token cache.

    Args:
      formatted_account_id: str, The formatted account id.

    Returns:
      tuple: The access_token, token_expiry, rapt_token, id_token tuple.
    """
        with self._cursor as cur:
            return cur.Execute('SELECT access_token, token_expiry, rapt_token, id_token FROM "{}" WHERE account_id = ?'.format(_ACCESS_TOKEN_TABLE), (formatted_account_id,)).fetchone()

    def Store(self, formatted_account_id, access_token, token_expiry, rapt_token, id_token):
        """Stores the tokens into the access token cache.

    Args:
      formatted_account_id: str, The formatted account id.
      access_token: str, The access token string to store.
      token_expiry: datetime.datetime, The token expiry.
      rapt_token: str, The rapt token string to store.
      id_token: str, The ID token string to store.
    """
        try:
            self._Execute('REPLACE INTO "{}" (account_id, access_token, token_expiry, rapt_token, id_token) VALUES (?,?,?,?,?)'.format(_ACCESS_TOKEN_TABLE), (formatted_account_id, access_token, token_expiry, rapt_token, id_token))
        except sqlite3.OperationalError as e:
            log.warning('Could not store access token in cache: {}'.format(str(e)))

    def Remove(self, formatted_account_id):
        """Removes the tokens from the access token cache.

    Args:
      formatted_account_id: str, The formatted account id to remove.
    """
        try:
            self._Execute('DELETE FROM "{}" WHERE account_id = ?'.format(_ACCESS_TOKEN_TABLE), (formatted_account_id,))
        except sqlite3.OperationalError as e:
            log.warning('Could not delete access token from cache: {}'.format(str(e)))