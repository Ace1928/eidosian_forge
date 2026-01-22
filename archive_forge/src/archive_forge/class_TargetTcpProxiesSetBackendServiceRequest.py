from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetTcpProxiesSetBackendServiceRequest(_messages.Message):
    """A TargetTcpProxiesSetBackendServiceRequest object.

  Fields:
    service: The URL of the new BackendService resource for the
      targetTcpProxy.
  """
    service = _messages.StringField(1)