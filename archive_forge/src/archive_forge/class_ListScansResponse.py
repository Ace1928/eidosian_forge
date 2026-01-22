from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListScansResponse(_messages.Message):
    """Response method from the ListScans method.

  Fields:
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
    scans: Available scans based on the list query parameters.
  """
    nextPageToken = _messages.StringField(1)
    scans = _messages.MessageField('Scan', 2, repeated=True)