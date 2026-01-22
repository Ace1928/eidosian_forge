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
def AcquireFromGCE(account=None, use_google_auth=True, refresh=True):
    """Get credentials from a GCE metadata server.

  Args:
    account: str, The account name to use. If none, the default is used.
    use_google_auth: bool, True to load credentials of google-auth if it is
      supported in the current authentication scenario. False to load
      credentials of oauth2client.
    refresh: bool, Whether to refresh the credential or not. The default value
      is True.

  Returns:
    oauth2client.client.Credentials or google.auth.credentials.Credentials based
    on use_google_auth and whether google-auth is supported in the current
    authentication sceanrio.

  Raises:
    c_gce.CannotConnectToMetadataServerException: If the metadata server cannot
      be reached.
    TokenRefreshError: If the credentials fail to refresh.
    TokenRefreshReauthError: If the credentials fail to refresh due to reauth.
  """
    if use_google_auth:
        import google.auth.compute_engine as google_auth_gce
        email = account or 'default'
        credentials = google_auth_gce.Credentials(service_account_email=email)
        universe_domain = c_gce.Metadata().UniverseDomain()
        credentials._universe_domain = universe_domain
        credentials._universe_domain_cached = True
    else:
        credentials = oauth2client_gce.AppAssertionCredentials(email=account)
    if refresh:
        Refresh(credentials)
    return credentials