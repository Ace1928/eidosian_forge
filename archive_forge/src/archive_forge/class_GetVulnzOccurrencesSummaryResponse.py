from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GetVulnzOccurrencesSummaryResponse(_messages.Message):
    """A summary of how many vulnz occurrences there are per severity type.
  counts by groups, or if we should have different summary messages like this.

  Fields:
    counts: A map of how many occurrences were found for each severity.
  """
    counts = _messages.MessageField('SeverityCount', 1, repeated=True)