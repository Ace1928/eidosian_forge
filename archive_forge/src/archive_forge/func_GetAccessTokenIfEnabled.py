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
def GetAccessTokenIfEnabled(account=None, scopes=None, allow_account_impersonation=True):
    """Returns the access token of the given account or the active account.

  If credentials have been disabled via properties, this will return None.
  Otherwise it return the access token of the account like normal. Use this
  function when credentials are optional for the caller, or the caller want to
  handle the situation of credentials being disabled by properties.

  Args:
    account: str, The account to get the access token for. If None, the
      account stored in the core.account property is used.
    scopes: tuple, Custom auth scopes to request. By default CLOUDSDK_SCOPES are
      requested.
    allow_account_impersonation: bool, True to allow use of impersonated service
      account credentials (if that is configured).
  """
    if properties.VALUES.auth.disable_credentials.GetBool():
        return None
    return GetAccessToken(account, scopes, allow_account_impersonation)