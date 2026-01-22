from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupsRemoveInstancesRequest(_messages.Message):
    """A InstanceGroupsRemoveInstancesRequest object.

  Fields:
    instances: The list of instances to remove from the instance group.
  """
    instances = _messages.MessageField('InstanceReference', 1, repeated=True)