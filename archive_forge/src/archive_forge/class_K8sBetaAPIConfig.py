from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class K8sBetaAPIConfig(_messages.Message):
    """K8sBetaAPIConfig , configuration for beta APIs

  Fields:
    enabledApis: Enabled k8s beta APIs.
  """
    enabledApis = _messages.StringField(1, repeated=True)