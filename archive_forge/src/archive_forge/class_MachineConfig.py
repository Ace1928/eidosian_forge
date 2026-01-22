from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MachineConfig(_messages.Message):
    """MachineConfig describes the configuration of a machine.

  Fields:
    cpuCount: The number of CPU's in the VM instance.
  """
    cpuCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)