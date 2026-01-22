from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import base
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.credentials import creds as core_creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import store
from googlecloudsdk.core.util import files
from oauth2client import client
import six
from google.auth import exceptions as google_auth_exceptions
def WrapCredentials(self, http_client, allow_account_impersonation=True, use_google_auth=None):
    """Get an http client for working with Google APIs.

    Args:
      http_client: The http client to be wrapped.
      allow_account_impersonation: bool, True to allow use of impersonated
        service account credentials for calls made with this client. If False,
        the active user credentials will always be used.
      use_google_auth: bool, True if the calling command indicates to use
        google-auth library for authentication. If False, authentication will
        fallback to using the oauth2client library. If None, set the value based
        the configuration.

    Returns:
      An authorized http client with exception handling.

    Raises:
      creds_exceptions.Error: If an error loading the credentials occurs.
    """
    authority_selector = properties.VALUES.auth.authority_selector.Get()
    authorization_token_file = properties.VALUES.auth.authorization_token_file.Get()
    handlers = _GetIAMAuthHandlers(authority_selector, authorization_token_file)
    if use_google_auth is None:
        use_google_auth = base.UseGoogleAuth()
    creds = store.LoadIfEnabled(allow_account_impersonation, use_google_auth)
    if creds:
        http_client = self.AuthorizeClient(http_client, creds)
        setattr(http_client, '_googlecloudsdk_credentials', creds)
    self.WrapRequest(http_client, handlers, _HandleAuthError, (client.AccessTokenRefreshError, google_auth_exceptions.RefreshError))
    return http_client