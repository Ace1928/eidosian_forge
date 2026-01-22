from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisGoogleDevtoolsCloudbuildV1Secrets(_messages.Message):
    """Secrets and secret environment variables.

  Fields:
    inline: Secrets encrypted with KMS key and the associated secret
      environment variable.
    secretManager: Secrets in Secret Manager and associated secret environment
      variable.
  """
    inline = _messages.MessageField('ContaineranalysisGoogleDevtoolsCloudbuildV1InlineSecret', 1, repeated=True)
    secretManager = _messages.MessageField('ContaineranalysisGoogleDevtoolsCloudbuildV1SecretManagerSecret', 2, repeated=True)