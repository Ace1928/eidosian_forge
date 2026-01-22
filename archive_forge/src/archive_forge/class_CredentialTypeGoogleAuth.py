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
class CredentialTypeGoogleAuth(enum.Enum):
    """Enum of google-auth credential types managed by gcloud."""
    UNKNOWN = (0, UNKNOWN_CREDS_NAME, False, False)
    USER_ACCOUNT = (1, USER_ACCOUNT_CREDS_NAME, True, True)
    SERVICE_ACCOUNT = (2, SERVICE_ACCOUNT_CREDS_NAME, True, False)
    P12_SERVICE_ACCOUNT = (3, P12_SERVICE_ACCOUNT_CREDS_NAME, False, False)
    DEVSHELL = (4, DEVSHELL_CREDS_NAME, True, True)
    GCE = (5, GCE_CREDS_NAME, True, False)
    IMPERSONATED_ACCOUNT = (6, IMPERSONATED_ACCOUNT_CREDS_NAME, True, False)
    EXTERNAL_ACCOUNT = (7, EXTERNAL_ACCOUNT_CREDS_NAME, True, False)
    EXTERNAL_ACCOUNT_USER = (8, EXTERNAL_ACCOUNT_USER_CREDS_NAME, True, True)
    EXTERNAL_ACCOUNT_AUTHORIZED_USER = (9, EXTERNAL_ACCOUNT_AUTHORIZED_USER_CREDS_NAME, True, True)

    def __init__(self, type_id, key, is_serializable, is_user):
        """Builds a credentials type instance given the credentials information.

    Args:
      type_id: string, ID for the credentials type, based on the enum constant
        value of the type.
      key: string, key of the credentials type, based on the enum constant value
        of the type.
      is_serializable: bool, whether the type of the credentials is
        serializable, based on the enum constant value of the type.
      is_user: bool, True if the credentials are of user account. False
        otherwise.

    Returns:
      CredentialTypeGoogleAuth, an instance of CredentialTypeGoogleAuth which
        is a gcloud internal representation of type of the google-auth
        credentials.
    """
        self.type_id = type_id
        self.key = key
        self.is_serializable = is_serializable
        self.is_user = is_user

    @staticmethod
    def FromTypeKey(key):
        """Returns the credentials type based on the input key."""
        for cred_type in CredentialTypeGoogleAuth:
            if cred_type.key == key:
                return cred_type
        return CredentialTypeGoogleAuth.UNKNOWN

    @staticmethod
    def FromCredentials(creds):
        """Returns the credentials type based on the input credentials."""
        if isinstance(creds, google_auth_compute_engine.Credentials):
            return CredentialTypeGoogleAuth.GCE
        if isinstance(creds, google_auth_impersonated.Credentials):
            return CredentialTypeGoogleAuth.IMPERSONATED_ACCOUNT
        if isinstance(creds, google_auth_external_account.Credentials) and (not creds.is_user):
            return CredentialTypeGoogleAuth.EXTERNAL_ACCOUNT
        if isinstance(creds, google_auth_external_account.Credentials) and creds.is_user:
            return CredentialTypeGoogleAuth.EXTERNAL_ACCOUNT_USER
        if isinstance(creds, google_auth_external_account_authorized_user.Credentials):
            return CredentialTypeGoogleAuth.EXTERNAL_ACCOUNT_AUTHORIZED_USER
        from google.oauth2 import service_account as google_auth_service_account
        from googlecloudsdk.core.credentials import p12_service_account as google_auth_p12_service_account
        if isinstance(creds, google_auth_p12_service_account.Credentials):
            return CredentialTypeGoogleAuth.P12_SERVICE_ACCOUNT
        if isinstance(creds, google_auth_service_account.Credentials):
            return CredentialTypeGoogleAuth.SERVICE_ACCOUNT
        if getattr(creds, 'refresh_token', None) is not None:
            return CredentialTypeGoogleAuth.USER_ACCOUNT
        return CredentialTypeGoogleAuth.UNKNOWN