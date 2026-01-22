from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1AuthConfigGoogleServiceAccountConfig(_messages.Message):
    """Config for Google Service Account Authentication.

  Fields:
    serviceAccount: Optional. The service account that the extension execution
      service runs as. - If the service account is specified, the
      `iam.serviceAccounts.getAccessToken` permission should be granted to
      Vertex AI Extension Service Agent (https://cloud.google.com/vertex-
      ai/docs/general/access-control#service-agents) on the specified service
      account. - If not specified, the Vertex AI Extension Service Agent will
      be used to execute the Extension.
  """
    serviceAccount = _messages.StringField(1)