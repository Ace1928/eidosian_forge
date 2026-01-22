from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import textwrap
from googlecloudsdk.command_lib.util import check_browser
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def DoInstalledAppBrowserFlowGoogleAuth(scopes, client_id_file=None, client_config=None, no_launch_browser=False, no_browser=False, remote_bootstrap=None, query_params=None, auth_proxy_redirect_uri=None):
    """Launches a 3LO oauth2 flow to get google-auth credentials.

  Args:
    scopes: [str], The list of scopes to authorize.
    client_id_file: str, The path to a file containing the client id and secret
      to use for the flow.  If None, the default client id for the Cloud SDK is
      used.
    client_config: Optional[Mapping], the client secrets and urls that should be
      used for the OAuth flow.
    no_launch_browser: bool, True if users specify --no-launch-browser flag to
      use the remote login with auth proxy flow.
    no_browser: bool, True if users specify --no-browser flag to ask another
      gcloud instance to help with authorization.
    remote_bootstrap: str, The auth parameters specified by --remote-bootstrap
      flag. Once used, it means the command is to help authorize another
      gcloud (i.e. gcloud without access to browser).
    query_params: Optional[Mapping], extra params to pass to the flow during
      `Run`. These params end up getting used as query
      params for authorization_url.
    auth_proxy_redirect_uri: str, The uri where OAuth service will redirect the
      user to once the authentication is complete for a remote login with auth
      proxy flow.
  Returns:
    core.credentials.google_auth_credentials.Credentials, The credentials
      obtained from the flow.
  """
    from google.auth import external_account_authorized_user
    from google.oauth2 import credentials as oauth2_credentials
    from googlecloudsdk.core.credentials import flow as c_flow
    if client_id_file:
        AssertClientSecretIsInstalledType(client_id_file)
    if not client_config:
        client_config = _CreateGoogleAuthClientConfig(client_id_file)
    if not query_params:
        query_params = {}
    can_launch_browser = check_browser.ShouldLaunchBrowser(attempt_launch_browser=True)
    if no_browser:
        user_creds = NoBrowserFlowRunner(scopes, client_config).Run(**query_params)
    elif remote_bootstrap:
        if not can_launch_browser:
            raise c_flow.WebBrowserInaccessible('Cannot launch browser. Please run this command on a machine where gcloud can launch a web browser.')
        user_creds = NoBrowserHelperRunner(scopes, client_config).Run(partial_auth_url=remote_bootstrap, **query_params)
    elif no_launch_browser or not can_launch_browser:
        user_creds = RemoteLoginWithAuthProxyFlowRunner(scopes, client_config, auth_proxy_redirect_uri).Run(**query_params)
    else:
        user_creds = BrowserFlowWithNoBrowserFallbackRunner(scopes, client_config).Run(**query_params)
    if user_creds:
        if isinstance(user_creds, oauth2_credentials.Credentials):
            from googlecloudsdk.core.credentials import google_auth_credentials as c_google_auth
            return c_google_auth.Credentials.FromGoogleAuthUserCredentials(user_creds)
        if isinstance(user_creds, external_account_authorized_user.Credentials):
            return user_creds