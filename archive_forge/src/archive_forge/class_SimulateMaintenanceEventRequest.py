from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SimulateMaintenanceEventRequest(_messages.Message):
    """Request for SimulateMaintenanceEvent.

  Fields:
    workerIds: The 0-based worker ID. If it is empty, worker ID 0 will be
      selected for maintenance event simulation. A maintenance event will only
      be fired on the first specified worker ID. Future implementations may
      support firing on multiple workers.
  """
    workerIds = _messages.StringField(1, repeated=True)