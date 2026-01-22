from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryProjectsListRequest(_messages.Message):
    """A BigqueryProjectsListRequest object.

  Fields:
    maxResults: Maximum number of results to return
    pageToken: Page token, returned by a previous call, to request the next
      page of results
  """
    maxResults = _messages.IntegerField(1, variant=_messages.Variant.UINT32)
    pageToken = _messages.StringField(2)