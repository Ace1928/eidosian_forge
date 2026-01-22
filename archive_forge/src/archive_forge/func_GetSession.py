from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from google.auth import external_account as google_auth_external_account
from google.auth.transport import requests as google_auth_requests
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import requests
from googlecloudsdk.core import transport as core_transport
from googlecloudsdk.core.credentials import transport
def GetSession(timeout='unset', ca_certs=None, enable_resource_quota=True, allow_account_impersonation=True, session=None, streaming_response_body=False, redact_request_body_reason=None):
    """Get requests.Session object for working with the Google API.

  Args:
    timeout: double, The timeout in seconds to pass to httplib2.  This is the
        socket level timeout.  If timeout is None, timeout is infinite.  If
        default argument 'unset' is given, a sensible default is selected.
    ca_certs: str, absolute filename of a ca_certs file that overrides the
        default
    enable_resource_quota: bool, By default, we are going to tell APIs to use
        the quota of the project being operated on. For some APIs we want to use
        gcloud's quota, so you can explicitly disable that behavior by passing
        False here.
    allow_account_impersonation: bool, True to allow use of impersonated service
        account credentials for calls made with this client. If False, the
        active user credentials will always be used.
    session: requests.Session instance. Otherwise, a new requests.Session will
        be initialized.
    streaming_response_body: bool, True indicates that the response body will
        be a streaming body.
    redact_request_body_reason: str, the reason why the request body must be
        redacted if --log-http is used. If None, the body is not redacted.

  Returns:
    1. A regular requests.Session object if no credentials are available;
    2. Or an authorized requests.Session object authorized by google-auth
       credentials.

  Raises:
    creds_exceptions.Error: If an error loading the credentials occurs.
  """
    session = requests.GetSession(timeout=timeout, ca_certs=ca_certs, session=session, streaming_response_body=streaming_response_body, redact_request_body_reason=redact_request_body_reason)
    request_wrapper = RequestWrapper()
    session = request_wrapper.WrapQuota(session, enable_resource_quota, allow_account_impersonation, True)
    session = request_wrapper.WrapCredentials(session, allow_account_impersonation)
    return session