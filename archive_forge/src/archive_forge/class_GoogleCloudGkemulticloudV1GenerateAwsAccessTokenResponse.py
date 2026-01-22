from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1GenerateAwsAccessTokenResponse(_messages.Message):
    """Response message for `AwsClusters.GenerateAwsAccessToken` method.

  Fields:
    accessToken: Output only. Access token to authenticate to k8s api-server.
    expirationTime: Output only. Timestamp at which the token will expire.
  """
    accessToken = _messages.StringField(1)
    expirationTime = _messages.StringField(2)