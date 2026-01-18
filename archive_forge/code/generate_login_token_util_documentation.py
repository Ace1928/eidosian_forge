from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from google.auth import credentials
from google.auth import exceptions as google_auth_exceptions
from google.oauth2 import credentials as google_auth_creds
from googlecloudsdk.api_lib.auth import exceptions as auth_exceptions
from googlecloudsdk.calliope import exceptions as c_exc
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import google_auth_credentials as c_google_auth
from googlecloudsdk.core.credentials import store as c_store
import six
Genearte a down-scoped credential.

  Args:
    creds: end user credential
    scopes: scopes to be included in the down-scoped credential

  Returns:
    Down-scoped credential.
  