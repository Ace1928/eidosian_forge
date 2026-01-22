from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManagersSetTargetPoolsRequest(_messages.Message):
    """A InstanceGroupManagersSetTargetPoolsRequest object.

  Fields:
    fingerprint: The fingerprint of the target pools information. Use this
      optional property to prevent conflicts when multiple users change the
      target pools settings concurrently. Obtain the fingerprint with the
      instanceGroupManagers.get method. Then, include the fingerprint in your
      request to ensure that you do not overwrite changes that were applied
      from another concurrent request.
    targetPools: The list of target pool URLs that instances in this managed
      instance group belong to. The managed instance group applies these
      target pools to all of the instances in the group. Existing instances
      and new instances in the group all receive these target pool settings.
  """
    fingerprint = _messages.BytesField(1)
    targetPools = _messages.StringField(2, repeated=True)