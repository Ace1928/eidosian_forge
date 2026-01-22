from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeTargetGrpcProxiesGetRequest(_messages.Message):
    """A ComputeTargetGrpcProxiesGetRequest object.

  Fields:
    project: Project ID for this request.
    targetGrpcProxy: Name of the TargetGrpcProxy resource to return.
  """
    project = _messages.StringField(1, required=True)
    targetGrpcProxy = _messages.StringField(2, required=True)