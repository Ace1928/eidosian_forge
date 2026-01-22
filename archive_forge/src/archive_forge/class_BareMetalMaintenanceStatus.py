from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalMaintenanceStatus(_messages.Message):
    """Represents the maintenance status of the bare metal user cluster.

  Fields:
    machineDrainStatus: The maintenance status of node machines.
  """
    machineDrainStatus = _messages.MessageField('BareMetalMachineDrainStatus', 1)