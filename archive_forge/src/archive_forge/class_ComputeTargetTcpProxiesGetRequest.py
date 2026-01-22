from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeTargetTcpProxiesGetRequest(_messages.Message):
    """A ComputeTargetTcpProxiesGetRequest object.

  Fields:
    project: Project ID for this request.
    targetTcpProxy: Name of the TargetTcpProxy resource to return.
  """
    project = _messages.StringField(1, required=True)
    targetTcpProxy = _messages.StringField(2, required=True)