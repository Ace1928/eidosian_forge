from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListFindingsResponse(_messages.Message):
    """Response for the `ListFindings` method.

  Fields:
    findings: The list of Findings returned.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    findings = _messages.MessageField('Finding', 1, repeated=True)
    nextPageToken = _messages.StringField(2)