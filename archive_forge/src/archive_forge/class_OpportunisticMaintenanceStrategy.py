from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OpportunisticMaintenanceStrategy(_messages.Message):
    """Strategy that will trigger maintenance on behalf of the customer.

  Fields:
    maintenanceAvailabilityWindow: The window of time that opportunistic
      maintenance can run. Example: A setting of 14 days implies that
      opportunistic maintenance can only be ran in the 2 weeks leading up to
      the scheduled maintenance date. Setting 28 days allows opportunistic
      maintenance to run at any time in the scheduled maintenance window (all
      `PERIODIC` maintenance is set 28 days in advance).
    minNodesPerPool: The minimum nodes required to be available in a pool.
      Blocks maintenance if it would cause the number of running nodes to dip
      below this value.
    nodeIdleTimeWindow: The amount of time that a node can remain idle (no
      customer owned workloads running), before triggering maintenance.
  """
    maintenanceAvailabilityWindow = _messages.StringField(1)
    minNodesPerPool = _messages.IntegerField(2)
    nodeIdleTimeWindow = _messages.StringField(3)