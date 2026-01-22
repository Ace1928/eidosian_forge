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
def _RefreshServiceAccountIdToken(cred, http_client):
    """Get a fresh id_token for the given oauth2client credentials.

  Args:
    cred: service_account.ServiceAccountCredentials, the credentials for which
      to refresh the id_token.
    http_client: httplib2.Http, the http transport to refresh with.

  Returns:
    str, The id_token if refresh was successful. Otherwise None.
  """
    http_request = http_client.request
    now = int(time.time())
    payload = {'aud': cred.token_uri, 'iat': now, 'exp': now + cred.MAX_TOKEN_LIFETIME_SECS, 'iss': cred._service_account_email, 'target_audience': config.CLOUDSDK_CLIENT_ID}
    assertion = crypt.make_signed_jwt(cred._signer, payload, key_id=cred._private_key_id)
    body = urllib.parse.urlencode({'assertion': assertion, 'grant_type': _GRANT_TYPE})
    resp, content = http_request(cred.token_uri.encode('idna'), method='POST', body=body, headers=cred._generate_refresh_request_headers())
    if resp.status == 200:
        d = json.loads(content)
        return d.get('id_token', None)
    else:
        return None