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
def FromJsonGoogleAuth(json_value):
    """Returns google-auth credentials from library independent json format.

  The type of the credentials could be service account, external account
  (workload identity pool or workforce pool), external account authorized user
  (workforce), user account, p12 service account, or compute engine.

  Args:
    json_value: string, A string of the JSON representation of the credentials.

  Returns:
    google.auth.credentials.Credentials if the credentials type is supported
    by this method.

  Raises:
    UnknownCredentialsType: when the type of the credentials is not service
      account, user account or external account.
    InvalidCredentialsError: when the provided credentials are malformed or
      unsupported external account credentials.
  """
    json_key = json.loads(json_value)
    cred_type = CredentialTypeGoogleAuth.FromTypeKey(json_key['type'])
    if cred_type == CredentialTypeGoogleAuth.SERVICE_ACCOUNT:
        json_key['token_uri'] = GetEffectiveTokenUri(json_key)
        from google.oauth2 import service_account as google_auth_service_account
        service_account_credentials = google_auth_service_account.Credentials.from_service_account_info
        cred = service_account_credentials(json_key, scopes=config.CLOUDSDK_SCOPES)
        cred.private_key = json_key.get('private_key')
        cred.private_key_id = json_key.get('private_key_id')
        cred.client_id = json_key.get('client_id')
        EnableSelfSignedJwtIfApplicable(cred)
        return cred
    if cred_type == CredentialTypeGoogleAuth.P12_SERVICE_ACCOUNT:
        json_key['token_uri'] = GetEffectiveTokenUri(json_key)
        from googlecloudsdk.core.credentials import p12_service_account
        cred = p12_service_account.CreateP12ServiceAccount(base64.b64decode(json_key['private_key']), json_key['password'], service_account_email=json_key['client_email'], token_uri=json_key['token_uri'], scopes=config.CLOUDSDK_SCOPES)
        return cred
    if cred_type == CredentialTypeGoogleAuth.EXTERNAL_ACCOUNT:
        if 'service_account_impersonation_url' not in json_key:
            json_key['client_id'] = config.CLOUDSDK_CLIENT_ID
            json_key['client_secret'] = config.CLOUDSDK_CLIENT_NOTSOSECRET
        try:
            if json_key.get('subject_token_type') == 'urn:ietf:params:aws:token-type:aws4_request':
                from google.auth import aws
                cred = aws.Credentials.from_info(json_key, scopes=config.CLOUDSDK_SCOPES)
            elif json_key.get('credential_source') is not None and json_key.get('credential_source').get('executable') is not None:
                from google.auth import pluggable
                executable = json_key.get('credential_source').get('executable')
                cred = pluggable.Credentials.from_info(json_key, scopes=config.CLOUDSDK_SCOPES)
                if cred.is_workforce_pool and executable.get('interactive_timeout_millis'):
                    cred.interactive = True
                    setattr(cred, '_tokeninfo_username', json_key.get('external_account_id') or '')
            else:
                from google.auth import identity_pool
                cred = identity_pool.Credentials.from_info(json_key, scopes=config.CLOUDSDK_SCOPES)
        except (ValueError, TypeError, google_auth_exceptions.RefreshError):
            raise InvalidCredentialsError('The provided external account credentials are invalid or unsupported')
        return WrapGoogleAuthExternalAccountRefresh(cred)
    if cred_type == CredentialTypeGoogleAuth.EXTERNAL_ACCOUNT_AUTHORIZED_USER:
        json_key['client_id'] = config.CLOUDSDK_CLIENT_ID
        json_key['client_secret'] = config.CLOUDSDK_CLIENT_NOTSOSECRET
        json_key['scopes'] = config.CLOUDSDK_EXTERNAL_ACCOUNT_SCOPES
        try:
            cred = google_auth_external_account_authorized_user.Credentials.from_info(json_key)
        except (ValueError, TypeError, google_auth_exceptions.RefreshError):
            raise InvalidCredentialsError('The provided external account authorized user credentials are invalid or unsupported')
        return WrapGoogleAuthExternalAccountRefresh(cred)
    if cred_type == CredentialTypeGoogleAuth.USER_ACCOUNT:
        json_key['token_uri'] = GetEffectiveTokenUri(json_key)
        from googlecloudsdk.core.credentials import google_auth_credentials as c_google_auth
        cred = c_google_auth.Credentials.from_authorized_user_info(json_key, scopes=json_key.get('scopes'))
        cred._token_uri = json_key['token_uri']
        return cred
    if cred_type == CredentialTypeGoogleAuth.GCE:
        cred = google_auth_compute_engine.Credentials(service_account_email=json_key['service_account_email'])
        cred._universe_domain = json_key.get('universe_domain', properties.VALUES.core.universe_domain.default)
        cred._universe_domain_cached = True
        return cred
    raise UnknownCredentialsType('Google auth does not support deserialization of {} credentials.'.format(json_key['type']))