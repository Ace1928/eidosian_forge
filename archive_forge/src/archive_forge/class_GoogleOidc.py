from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleOidc(_messages.Message):
    """Represents a config used to authenticate with a Google OIDC token using
  a GCP service account. Use this authentication method to invoke your Cloud
  Run and Cloud Functions destinations or HTTP endpoints that support Google
  OIDC.

  Fields:
    audience: Optional. Audience to be used to generate the OIDC Token. The
      audience claim identifies the recipient that the JWT is intended for. If
      unspecified, the destination URI will be used.
    serviceAccount: Required. The IAM service account email used as the
      identity of the stream resource. The service account is used to generate
      OIDC tokens for the outbound messages. It's also used to read messages
      from the "source". In addition to service account email, the resource
      name of the service account can be used in the format of
      `projects/-/serviceAccounts/{ACCOUNT}`, ACCOUNT can be email address or
      uniqueId of the service account (see https://cloud.google.com/iam/refere
      nce/rest/v1/projects.serviceAccounts/get).
  """
    audience = _messages.StringField(1)
    serviceAccount = _messages.StringField(2)