from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from google.auth import external_account as google_auth_external_account
import google_auth_httplib2
from googlecloudsdk.calliope import base
from googlecloudsdk.core import http
from googlecloudsdk.core.credentials import creds as core_creds
from googlecloudsdk.core.credentials import transport
import six
def Http(timeout='unset', response_encoding=None, ca_certs=None, enable_resource_quota=True, allow_account_impersonation=True, use_google_auth=None):
    """Get an httplib2.Http client for working with the Google API.

  Args:
    timeout: double, The timeout in seconds to pass to httplib2.  This is the
        socket level timeout.  If timeout is None, timeout is infinite.  If
        default argument 'unset' is given, a sensible default is selected.
    response_encoding: str, the encoding to use to decode the response.
    ca_certs: str, absolute filename of a ca_certs file that overrides the
        default
    enable_resource_quota: bool, By default, we are going to tell APIs to use
        the quota of the project being operated on. For some APIs we want to use
        gcloud's quota, so you can explicitly disable that behavior by passing
        False here.
    allow_account_impersonation: bool, True to allow use of impersonated service
      account credentials for calls made with this client. If False, the active
      user credentials will always be used.
    use_google_auth: bool, True if the calling command indicates to use
      google-auth library for authentication. If False, authentication will
      fallback to using the oauth2client library. If None, set the value based
      on the configuration.

  Returns:
    1. A regular httplib2.Http object if no credentials are available;
    2. Or a httplib2.Http client object authorized by oauth2client
       credentials if use_google_auth==False;
    3. Or a google_auth_httplib2.AuthorizedHttp client object authorized by
       google-auth credentials.

  Raises:
    core.credentials.exceptions.Error: If an error loading the credentials
      occurs.
  """
    http_client = http.Http(timeout=timeout, response_encoding=response_encoding, ca_certs=ca_certs)
    if use_google_auth is None:
        use_google_auth = base.UseGoogleAuth()
    request_wrapper = RequestWrapper()
    http_client = request_wrapper.WrapQuota(http_client, enable_resource_quota, allow_account_impersonation, use_google_auth)
    http_client = request_wrapper.WrapCredentials(http_client, allow_account_impersonation, use_google_auth)
    if hasattr(http_client, '_googlecloudsdk_credentials'):
        creds = http_client._googlecloudsdk_credentials
        if core_creds.IsGoogleAuthCredentials(creds):
            apitools_creds = _GoogleAuthApitoolsCredentials(creds)
        else:
            apitools_creds = creds
        setattr(http_client.request, 'credentials', apitools_creds)
    return http_client