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
def ToDictGoogleAuth(credentials):
    """Given google-auth credentials, recursively return dict representation.

  This method is added because google-auth credentials are not serializable
  natively.

  Args:
    credentials: google-auth credential object.

  Returns:
    Dict representation of the credential.

  Raises:
    UnknownCredentialsType: An error for when we fail to determine the type
    of the credentials.
  """
    creds_type = CredentialTypeGoogleAuth.FromCredentials(credentials)
    if not creds_type.is_serializable:
        raise UnknownCredentialsType('Google auth does not support serialization of {} credentials.'.format(creds_type.key))
    creds_dict = {'type': creds_type.key}
    filtered_list = [attr for attr in dir(credentials) if not attr.startswith('__') and attr not in ['signer', '_abc_negative_cache_version']]
    attr_list = [attr for attr in filtered_list if not attr.startswith('_') or attr[1:] not in filtered_list]
    attr_list = sorted(attr_list)
    for attr in attr_list:
        if hasattr(credentials, attr):
            val = getattr(credentials, attr)
            val_type = type(val)
            if val_type == datetime.datetime:
                val = val.strftime('%m-%d-%Y %H:%M:%S')
            elif issubclass(val_type, google_auth_creds.Credentials):
                try:
                    val = ToDictGoogleAuth(val)
                except UnknownCredentialsType:
                    continue
            elif val is not None and (not isinstance(val, six.string_types)) and (val_type not in (int, float, bool, str, list, dict, tuple)):
                continue
            creds_dict[attr] = val
    return creds_dict