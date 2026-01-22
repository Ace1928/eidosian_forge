from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListControlReportsResponse(_messages.Message):
    """Response message with all the control reports.

  Fields:
    controlReports: Output only. The control reports.
    nextPageToken: Output only. The token to retrieve the next page of
      results.
  """
    controlReports = _messages.MessageField('ControlReport', 1, repeated=True)
    nextPageToken = _messages.StringField(2)