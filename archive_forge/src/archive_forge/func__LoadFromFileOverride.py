from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import datetime
import json
import os
import textwrap
import time
from typing import Optional
import dateutil
from googlecloudsdk.api_lib.auth import util as auth_util
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import gce as c_gce
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
from oauth2client import client
from oauth2client import crypt
from oauth2client import service_account
from oauth2client.contrib import gce as oauth2client_gce
from oauth2client.contrib import reauth_errors
import six
from six.moves import urllib
def _LoadFromFileOverride(cred_file_override, scopes, use_google_auth):
    """Load credentials from cred file override."""
    log.info('Using alternate credentials from file: [%s]', cred_file_override)
    if not use_google_auth:
        try:
            cred = client.GoogleCredentials.from_stream(cred_file_override)
        except client.Error as e:
            raise InvalidCredentialFileException(cred_file_override, e)
        if cred.create_scoped_required():
            if scopes is None:
                scopes = config.CLOUDSDK_SCOPES
            cred = cred.create_scoped(scopes)
        token_uri_override = properties.VALUES.auth.token_host.Get()
        if token_uri_override:
            cred_type = c_creds.CredentialType.FromCredentials(cred)
            if cred_type in (c_creds.CredentialType.SERVICE_ACCOUNT, c_creds.CredentialType.P12_SERVICE_ACCOUNT):
                cred.token_uri = token_uri_override
        cred = c_creds.MaybeAttachAccessTokenCacheStore(cred)
    else:
        google_auth_default = c_creds.GetGoogleAuthDefault()
        from google.auth import credentials as google_auth_creds
        from google.auth import exceptions as google_auth_exceptions
        from google.auth import external_account as google_auth_external_account
        from google.auth import external_account_authorized_user as google_auth_external_account_authorized_user
        try:
            cred, _ = google_auth_default.load_credentials_from_file(cred_file_override)
        except google_auth_exceptions.DefaultCredentialsError as e:
            raise InvalidCredentialFileException(cred_file_override, e)
        if scopes is None:
            scopes = config.CLOUDSDK_SCOPES
        cred = google_auth_creds.with_scopes_if_required(cred, scopes)
        if isinstance(cred, google_auth_external_account.Credentials) and (not cred.service_account_email):
            json_info = cred.info
            json_info['client_id'] = config.CLOUDSDK_CLIENT_ID
            json_info['client_secret'] = config.CLOUDSDK_CLIENT_NOTSOSECRET
            cred = type(cred).from_info(json_info, scopes=config.CLOUDSDK_SCOPES)
        if isinstance(cred, google_auth_external_account_authorized_user.Credentials):
            json_info = cred.info
            json_info['client_id'] = config.CLOUDSDK_CLIENT_ID
            json_info['client_secret'] = config.CLOUDSDK_CLIENT_NOTSOSECRET
            json_info['scopes'] = config.CLOUDSDK_EXTERNAL_ACCOUNT_SCOPES
            cred = type(cred).from_info(json_info)
        cred_type = c_creds.CredentialTypeGoogleAuth.FromCredentials(cred)
        if cred_type == c_creds.CredentialTypeGoogleAuth.SERVICE_ACCOUNT:
            token_uri_override = properties.VALUES.auth.token_host.Get()
            if token_uri_override:
                cred._token_uri = token_uri_override
        elif cred_type == c_creds.CredentialTypeGoogleAuth.USER_ACCOUNT:
            token_uri_override = auth_util.GetTokenUri()
            cred._token_uri = token_uri_override
        c_creds.EnableSelfSignedJwtIfApplicable(cred)
        cred = c_creds.MaybeAttachAccessTokenCacheStoreGoogleAuth(cred)
    return cred