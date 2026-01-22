from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1AuthConfigOidcConfig(_messages.Message):
    """Config for user OIDC auth.

  Fields:
    idToken: OpenID Connect formatted ID token for extension endpoint. Only
      used to propagate token from
      [[ExecuteExtensionRequest.runtime_auth_config]] at request time.
    serviceAccount: The service account used to generate an OpenID Connect
      (OIDC)-compatible JWT token signed by the Google OIDC Provider
      (accounts.google.com) for extension endpoint
      (https://cloud.google.com/iam/docs/create-short-lived-credentials-
      direct#sa-credentials-oidc). - The audience for the token will be set to
      the URL in the server url defined in the OpenApi spec. - If the service
      account is provided, the service account should grant
      `iam.serviceAccounts.getOpenIdToken` permission to Vertex AI Extension
      Service Agent (https://cloud.google.com/vertex-ai/docs/general/access-
      control#service-agents).
  """
    idToken = _messages.StringField(1)
    serviceAccount = _messages.StringField(2)