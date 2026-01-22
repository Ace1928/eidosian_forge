from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManagerInstanceFlexibilityPolicyInstanceSelection(_messages.Message):
    """A InstanceGroupManagerInstanceFlexibilityPolicyInstanceSelection object.

  Fields:
    machineTypes: Full machine-type names, e.g. "n1-standard-16".
    rank: Preference of this instance selection. Lower number means higher
      preference. MIG will first try to create a VM based on the machine-type
      with lowest rank and fallback to next rank based on availability.
      Machine types and instance selections with the same rank have the same
      preference.
  """
    machineTypes = _messages.StringField(1, repeated=True)
    rank = _messages.IntegerField(2, variant=_messages.Variant.INT32)