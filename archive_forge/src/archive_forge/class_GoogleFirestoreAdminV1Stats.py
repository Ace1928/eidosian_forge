from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1Stats(_messages.Message):
    """Backup specific statistics.

  Fields:
    documentCount: Output only. The total number of documents contained in the
      backup.
    indexCount: Output only. The total number of index entries contained in
      the backup.
    sizeBytes: Output only. Summation of the size of all documents and index
      entries in the backup, measured in bytes.
  """
    documentCount = _messages.IntegerField(1)
    indexCount = _messages.IntegerField(2)
    sizeBytes = _messages.IntegerField(3)