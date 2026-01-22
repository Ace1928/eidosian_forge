from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalStandaloneMachineDrainStatus(_messages.Message):
    """Represents the status of node machines that are undergoing drain
  operations.

  Fields:
    drainedMachines: The list of drained machines.
    drainingMachines: The list of draning machines.
  """
    drainedMachines = _messages.MessageField('BareMetalStandaloneDrainedMachine', 1, repeated=True)
    drainingMachines = _messages.MessageField('BareMetalStandaloneDrainingMachine', 2, repeated=True)