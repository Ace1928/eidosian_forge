from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TableDataList(_messages.Message):
    """A TableDataList object.

  Fields:
    etag: A hash of this page of results.
    kind: The resource type of the response.
    pageToken: A token used for paging results. Providing this token instead
      of the startIndex parameter can help you retrieve stable results when an
      underlying table is changing.
    rows: Rows of results.
    totalRows: The total number of rows in the complete table.
  """
    etag = _messages.StringField(1)
    kind = _messages.StringField(2, default=u'bigquery#tableDataList')
    pageToken = _messages.StringField(3)
    rows = _messages.MessageField('TableRow', 4, repeated=True)
    totalRows = _messages.IntegerField(5)