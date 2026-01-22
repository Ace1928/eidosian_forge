from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AwsServicesAuthentication(_messages.Message):
    """Authentication configuration for the management of AWS resources.

  Fields:
    roleArn: Required. The Amazon Resource Name (ARN) of the role that the
      Anthos Multi-Cloud API will assume when managing AWS resources on your
      account.
    roleSessionName: Optional. An identifier for the assumed role session.
      When unspecified, it defaults to `multicloud-service-agent`.
  """
    roleArn = _messages.StringField(1)
    roleSessionName = _messages.StringField(2)