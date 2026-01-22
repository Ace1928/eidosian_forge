from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FeatureUpdate(_messages.Message):
    """Feature config to use for Rollout.

  Fields:
    binaryAuthorizationConfig: Optional. Configuration for Binary
      Authorization.
    securityPostureConfig: Optional. Configuration for Security Posture.
  """
    binaryAuthorizationConfig = _messages.MessageField('BinaryAuthorizationConfig', 1)
    securityPostureConfig = _messages.MessageField('SecurityPostureConfig', 2)