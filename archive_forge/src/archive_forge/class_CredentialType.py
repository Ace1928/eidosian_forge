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
class CredentialType(enum.Enum):
    """Enum of oauth2client credential types managed by gcloud."""
    UNKNOWN = (0, UNKNOWN_CREDS_NAME, False, False)
    USER_ACCOUNT = (1, USER_ACCOUNT_CREDS_NAME, True, True)
    SERVICE_ACCOUNT = (2, SERVICE_ACCOUNT_CREDS_NAME, True, False)
    P12_SERVICE_ACCOUNT = (3, P12_SERVICE_ACCOUNT_CREDS_NAME, True, False)
    DEVSHELL = (4, DEVSHELL_CREDS_NAME, False, True)
    GCE = (5, GCE_CREDS_NAME, False, False)

    def __init__(self, type_id, key, is_serializable, is_user):
        self.type_id = type_id
        self.key = key
        self.is_serializable = is_serializable
        self.is_user = is_user

    @staticmethod
    def FromTypeKey(key):
        for cred_type in CredentialType:
            if cred_type.key == key:
                return cred_type
        return CredentialType.UNKNOWN

    @staticmethod
    def FromCredentials(creds):
        if isinstance(creds, oauth2client_gce.AppAssertionCredentials):
            return CredentialType.GCE
        if isinstance(creds, service_account.ServiceAccountCredentials):
            if getattr(creds, '_private_key_pkcs12', None) is not None:
                return CredentialType.P12_SERVICE_ACCOUNT
            return CredentialType.SERVICE_ACCOUNT
        if getattr(creds, 'refresh_token', None) is not None:
            return CredentialType.USER_ACCOUNT
        return CredentialType.UNKNOWN