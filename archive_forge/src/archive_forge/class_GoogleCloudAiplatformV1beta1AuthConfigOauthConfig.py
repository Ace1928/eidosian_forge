from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1AuthConfigOauthConfig(_messages.Message):
    """Config for user oauth.

  Fields:
    accessToken: Access token for extension endpoint. Only used to propagate
      token from [[ExecuteExtensionRequest.runtime_auth_config]] at request
      time.
    serviceAccount: The service account used to generate access tokens for
      executing the Extension. - If the service account is specified, the
      `iam.serviceAccounts.getAccessToken` permission should be granted to
      Vertex AI Extension Service Agent (https://cloud.google.com/vertex-
      ai/docs/general/access-control#service-agents) on the provided service
      account.
  """
    accessToken = _messages.StringField(1)
    serviceAccount = _messages.StringField(2)