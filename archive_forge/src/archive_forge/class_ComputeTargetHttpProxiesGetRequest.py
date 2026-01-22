from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeTargetHttpProxiesGetRequest(_messages.Message):
    """A ComputeTargetHttpProxiesGetRequest object.

  Fields:
    project: Project ID for this request.
    targetHttpProxy: Name of the TargetHttpProxy resource to return.
  """
    project = _messages.StringField(1, required=True)
    targetHttpProxy = _messages.StringField(2, required=True)