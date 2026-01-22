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
def AcquireFromToken(refresh_token, token_uri=None, revoke_uri=GOOGLE_OAUTH2_PROVIDER_REVOKE_URI, use_google_auth=True):
    """Get credentials from an already-valid refresh token.

  Args:
    refresh_token: An oauth2 refresh token.
    token_uri: str, URI to use for refreshing.
    revoke_uri: str, URI to use for revoking.
    use_google_auth: bool, True to return google-auth credentials. False to
    return oauth2client credentials..

  Returns:
    oauth2client.client.Credentials or google.auth.credentials.Credentials.
    When all of the following conditions are true, it returns
    google.auth.credentials.Credentials and otherwise it returns
    oauth2client.client.Credentials.

    * use_google_auth=True
    * google-auth is not globally disabled by auth/disable_load_google_auth.
  """
    use_google_auth = use_google_auth and (not GoogleAuthDisabledGlobally())
    if token_uri is None:
        token_uri = auth_util.GetTokenUri()
    if use_google_auth:
        from google.oauth2 import credentials as google_auth_creds
        cred = google_auth_creds.Credentials(token=None, refresh_token=refresh_token, id_token=None, token_uri=token_uri, client_id=properties.VALUES.auth.client_id.Get(required=True), client_secret=properties.VALUES.auth.client_secret.Get(required=True))
        cred.expiry = datetime.datetime.utcnow()
    else:
        cred = client.OAuth2Credentials(access_token=None, client_id=properties.VALUES.auth.client_id.Get(required=True), client_secret=properties.VALUES.auth.client_secret.Get(required=True), refresh_token=refresh_token, token_expiry=datetime.datetime.utcnow(), token_uri=token_uri, user_agent=config.CLOUDSDK_USER_AGENT, revoke_uri=revoke_uri)
    return cred