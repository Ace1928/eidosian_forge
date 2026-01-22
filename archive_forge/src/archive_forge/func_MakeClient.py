from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from google.auth import external_account as google_auth_external_account
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import requests
from googlecloudsdk.core.credentials import creds
from googlecloudsdk.core.credentials import store
def MakeClient(client_class, credentials, address_override_func=None, mtls_enabled=False, attempt_direct_path=False):
    """Instantiates a gapic API client with gcloud defaults and configuration.

  grpc cannot be packaged like our other Python dependencies, due to platform
  differences and must be installed by the user. googlecloudsdk.core.gapic
  depends on grpc and must be imported lazily here so that this module can be
  imported safely anywhere.

  Args:
    client_class: a gapic client class.
    credentials: google.auth.credentials.Credentials, the credentials to use.
    address_override_func: function, function to call to override the client
      host. It takes a single argument which is the original host.
    mtls_enabled: bool, True if mTLS is enabled for this client.
    attempt_direct_path: bool, True if we want to attempt direct path gRPC where
      possible

  Returns:
    A gapic API client.
  """
    from googlecloudsdk.core import gapic_util_internal
    return client_class(transport=gapic_util_internal.MakeTransport(client_class, credentials, address_override_func, mtls_enabled, attempt_direct_path))