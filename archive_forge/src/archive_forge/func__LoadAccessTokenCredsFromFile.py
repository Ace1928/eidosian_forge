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
def _LoadAccessTokenCredsFromFile(token_file, use_google_auth):
    """Loads an AccessTokenCredentials from token_file."""
    log.info('Using access token from file: [%s]', token_file)
    if not use_google_auth:
        raise UnsupportedCredentialsError('You may have passed an access token to gcloud using --access-token-file or auth/access_token_file. At the same time, google-auth is disabled by auth/disable_load_google_auth. They do not work together. Please unset auth/disable_load_google_auth and retry.')
    content = files.ReadFileContents(token_file).strip()
    from googlecloudsdk.core.credentials import google_auth_credentials as c_google_auth
    creds = c_google_auth.AccessTokenCredentials(content)
    creds._universe_domain = properties.VALUES.core.universe_domain.Get()
    return creds