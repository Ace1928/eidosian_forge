from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import http
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import retry
from oauth2client import client as oauth2client_client
from oauth2client.contrib import reauth
import six
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import credentials as google_auth_credentials
from google.auth import external_account_authorized_user as google_auth_external_account_authorized_user
from google.auth import exceptions as google_auth_exceptions
from google.oauth2 import _client as google_auth_client
from google.oauth2 import credentials
from google.oauth2 import reauth as google_auth_reauth
@classmethod
def FromGoogleAuthUserCredentials(cls, creds):
    """Creates an object from creds of google.oauth2.credentials.Credentials.

    Args:
      creds: Union[
          google.oauth2.credentials.Credentials,
          google.auth.external_account_authorized_user.Credentials
      ], The input credentials.
    Returns:
      Credentials of Credentials.
    """
    if isinstance(creds, credentials.Credentials):
        res = cls(creds.token, refresh_token=creds.refresh_token, id_token=creds.id_token, token_uri=creds.token_uri, client_id=creds.client_id, client_secret=creds.client_secret, scopes=creds.scopes, quota_project_id=creds.quota_project_id)
        res.expiry = creds.expiry
        return res
    if isinstance(creds, google_auth_external_account_authorized_user.Credentials):
        return cls(creds.token, expiry=creds.expiry, refresh_token=creds.refresh_token, token_uri=creds.token_url, client_id=creds.client_id, client_secret=creds.client_secret, scopes=creds.scopes, quota_project_id=creds.quota_project_id)
    raise exceptions.InvalidCredentials('Invalid Credentials')