from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class FindingTypeStats(_messages.Message):
    """A FindingTypeStats resource represents stats regarding a specific
  FindingType of Findings under a given ScanRun.

  Fields:
    findingCount: The count of findings belonging to this finding type.
    findingType: The finding type associated with the stats.
  """
    findingCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    findingType = _messages.StringField(2)