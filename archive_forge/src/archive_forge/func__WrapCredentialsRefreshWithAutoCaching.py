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