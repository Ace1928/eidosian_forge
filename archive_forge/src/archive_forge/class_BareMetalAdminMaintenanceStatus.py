from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalAdminMaintenanceStatus(_messages.Message):
    """BareMetalAdminMaintenanceStatus represents the maintenance status for
  bare metal Admin cluster CR's nodes.

  Fields:
    machineDrainStatus: Represents the status of draining and drained machine
      nodes. This is used to show the progress of cluster upgrade.
  """
    machineDrainStatus = _messages.MessageField('BareMetalAdminMachineDrainStatus', 1)