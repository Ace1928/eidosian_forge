from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryTimelineSample(_messages.Message):
    """Summary of the state of query execution at a given time.

  Fields:
    activeUnits: Total number of active workers. This does not correspond
      directly to slot usage. This is the largest value observed since the
      last sample.
    completedUnits: Total parallel units of work completed by this query.
    elapsedMs: Milliseconds elapsed since the start of query execution.
    estimatedRunnableUnits: Units of work that can be scheduled immediately.
      Providing additional slots for these units of work will accelerate the
      query, if no other query in the reservation needs additional slots.
    pendingUnits: Total units of work remaining for the query. This number can
      be revised (increased or decreased) while the query is running.
    totalSlotMs: Cumulative slot-ms consumed by the query.
  """
    activeUnits = _messages.IntegerField(1)
    completedUnits = _messages.IntegerField(2)
    elapsedMs = _messages.IntegerField(3)
    estimatedRunnableUnits = _messages.IntegerField(4)
    pendingUnits = _messages.IntegerField(5)
    totalSlotMs = _messages.IntegerField(6)