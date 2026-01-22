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