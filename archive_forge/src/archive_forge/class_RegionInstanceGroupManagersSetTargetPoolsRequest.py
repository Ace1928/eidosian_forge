from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegionInstanceGroupManagersSetTargetPoolsRequest(_messages.Message):
    """A RegionInstanceGroupManagersSetTargetPoolsRequest object.

  Fields:
    fingerprint: Fingerprint of the target pools information, which is a hash
      of the contents. This field is used for optimistic locking when you
      update the target pool entries. This field is optional.
    targetPools: The URL of all TargetPool resources to which instances in the
      instanceGroup field are added. The target pools automatically apply to
      all of the instances in the managed instance group.
  """
    fingerprint = _messages.BytesField(1)
    targetPools = _messages.StringField(2, repeated=True)