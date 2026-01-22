from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReplicationStateValueValuesEnum(_messages.Enum):
    """Output only. The state of replication for the table in this cluster.

    Values:
      STATE_NOT_KNOWN: The replication state of the table is unknown in this
        cluster.
      INITIALIZING: The cluster was recently created, and the table must
        finish copying over pre-existing data from other clusters before it
        can begin receiving live replication updates and serving Data API
        requests.
      PLANNED_MAINTENANCE: The table is temporarily unable to serve Data API
        requests from this cluster due to planned internal maintenance.
      UNPLANNED_MAINTENANCE: The table is temporarily unable to serve Data API
        requests from this cluster due to unplanned or emergency maintenance.
      READY: The table can serve Data API requests from this cluster.
        Depending on replication delay, reads may not immediately reflect the
        state of the table in other clusters.
      READY_OPTIMIZING: The table is fully created and ready for use after a
        restore, and is being optimized for performance. When optimizations
        are complete, the table will transition to `READY` state.
    """
    STATE_NOT_KNOWN = 0
    INITIALIZING = 1
    PLANNED_MAINTENANCE = 2
    UNPLANNED_MAINTENANCE = 3
    READY = 4
    READY_OPTIMIZING = 5