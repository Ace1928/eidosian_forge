from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeRegionInstanceGroupManagersDeletePerInstanceConfigsRequest(_messages.Message):
    """A ComputeRegionInstanceGroupManagersDeletePerInstanceConfigsRequest
  object.

  Fields:
    instanceGroupManager: The name of the managed instance group. It should
      conform to RFC1035.
    project: Project ID for this request.
    region: Name of the region scoping this request, should conform to
      RFC1035.
    regionInstanceGroupManagerDeleteInstanceConfigReq: A
      RegionInstanceGroupManagerDeleteInstanceConfigReq resource to be passed
      as the request body.
  """
    instanceGroupManager = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    region = _messages.StringField(3, required=True)
    regionInstanceGroupManagerDeleteInstanceConfigReq = _messages.MessageField('RegionInstanceGroupManagerDeleteInstanceConfigReq', 4)