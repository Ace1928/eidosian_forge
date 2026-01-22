from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CdcStrategy(_messages.Message):
    """The strategy that the stream uses for CDC replication.

  Fields:
    mostRecentStartPosition: Optional. Start replicating from the most recent
      position in the source.
    nextAvailableStartPosition: Optional. Resume replication from the next
      available position in the source.
    specificStartPosition: Optional. Start replicating from a specific
      position in the source.
  """
    mostRecentStartPosition = _messages.MessageField('MostRecentStartPosition', 1)
    nextAvailableStartPosition = _messages.MessageField('NextAvailableStartPosition', 2)
    specificStartPosition = _messages.MessageField('SpecificStartPosition', 3)