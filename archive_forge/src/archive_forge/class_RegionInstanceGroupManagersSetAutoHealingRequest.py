from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegionInstanceGroupManagersSetAutoHealingRequest(_messages.Message):
    """A RegionInstanceGroupManagersSetAutoHealingRequest object.

  Fields:
    autoHealingPolicies: A InstanceGroupManagerAutoHealingPolicy attribute.
  """
    autoHealingPolicies = _messages.MessageField('InstanceGroupManagerAutoHealingPolicy', 1, repeated=True)