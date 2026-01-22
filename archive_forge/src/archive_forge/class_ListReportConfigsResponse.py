from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListReportConfigsResponse(_messages.Message):
    """Message for response to listing ReportConfigs

  Fields:
    nextPageToken: A token identifying a page of results the server should
      return.
    reportConfigs: The list of ReportConfig
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    reportConfigs = _messages.MessageField('ReportConfig', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)