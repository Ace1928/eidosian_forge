from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstancesSetMachineResourcesRequest(_messages.Message):
    """A InstancesSetMachineResourcesRequest object.

  Fields:
    guestAccelerators: A list of the type and count of accelerator cards
      attached to the instance.
  """
    guestAccelerators = _messages.MessageField('AcceleratorConfig', 1, repeated=True)