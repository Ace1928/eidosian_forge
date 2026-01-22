from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1ImportEntriesResponse(_messages.Message):
    """Response message for long-running operation returned by the
  ImportEntries.

  Fields:
    deletedEntriesCount: Number of entries deleted as a result of import
      operation.
    upsertedEntriesCount: Cumulative number of entries created and entries
      updated as a result of import operation.
  """
    deletedEntriesCount = _messages.IntegerField(1)
    upsertedEntriesCount = _messages.IntegerField(2)