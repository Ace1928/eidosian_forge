from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnvFromSource(_messages.Message):
    """Not supported by Cloud Run. EnvFromSource represents the source of a set
  of ConfigMaps

  Fields:
    configMapRef: The ConfigMap to select from
    prefix: An optional identifier to prepend to each key in the ConfigMap.
      Must be a C_IDENTIFIER.
    secretRef: The Secret to select from
  """
    configMapRef = _messages.MessageField('ConfigMapEnvSource', 1)
    prefix = _messages.StringField(2)
    secretRef = _messages.MessageField('SecretEnvSource', 3)