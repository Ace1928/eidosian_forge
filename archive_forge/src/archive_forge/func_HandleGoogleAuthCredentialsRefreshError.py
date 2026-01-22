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
@contextlib.contextmanager
def HandleGoogleAuthCredentialsRefreshError(for_adc=False):
    """Handles exceptions during refreshing google auth credentials."""
    from google.auth import exceptions as google_auth_exceptions
    from googlecloudsdk.core import context_aware
    from googlecloudsdk.core.credentials import google_auth_credentials as c_google_auth
    try:
        yield
    except reauth_errors.ReauthSamlLoginRequiredError:
        raise creds_exceptions.WebLoginRequiredReauthError(for_adc=for_adc)
    except (reauth_errors.ReauthError, c_google_auth.ReauthRequiredError) as e:
        raise creds_exceptions.TokenRefreshReauthError(str(e), for_adc=for_adc)
    except google_auth_exceptions.RefreshError as e:
        if context_aware.IsContextAwareAccessDeniedError(e):
            raise creds_exceptions.TokenRefreshDeniedByCAAError(e)
        raise creds_exceptions.TokenRefreshError(six.text_type(e), for_adc=for_adc)