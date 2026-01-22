from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1GenerateAzureClusterAgentTokenRequest(_messages.Message):
    """A GoogleCloudGkemulticloudV1GenerateAzureClusterAgentTokenRequest
  object.

  Fields:
    audience: Optional.
    grantType: Optional.
    nodePoolId: Optional.
    options: Optional.
    requestedTokenType: Optional.
    scope: Optional.
    subjectToken: Required.
    subjectTokenType: Required.
    version: Required.
  """
    audience = _messages.StringField(1)
    grantType = _messages.StringField(2)
    nodePoolId = _messages.StringField(3)
    options = _messages.StringField(4)
    requestedTokenType = _messages.StringField(5)
    scope = _messages.StringField(6)
    subjectToken = _messages.StringField(7)
    subjectTokenType = _messages.StringField(8)
    version = _messages.StringField(9)