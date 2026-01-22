from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmCreationConfig(_messages.Message):
    """VM creation configuration message.

  Fields:
    subnet: The subnet name the vm needs to be created in.
    vmMachineType: Required. VM instance machine type to create.
    vmZone: The Google Cloud Platform zone to create the VM in.
    vpc: The VPC name the vm needs to be created in.
  """
    subnet = _messages.StringField(1)
    vmMachineType = _messages.StringField(2)
    vmZone = _messages.StringField(3)
    vpc = _messages.StringField(4)