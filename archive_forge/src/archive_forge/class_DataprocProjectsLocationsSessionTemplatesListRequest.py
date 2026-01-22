from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsSessionTemplatesListRequest(_messages.Message):
    """A DataprocProjectsLocationsSessionTemplatesListRequest object.

  Fields:
    filter: Optional. A filter for the session templates to return in the
      response. Filters are case sensitive and have the following syntax:field
      = value AND field = value ...
    pageSize: Optional. The maximum number of sessions to return in each
      response. The service may return fewer than this value.
    pageToken: Optional. A page token received from a previous ListSessions
      call. Provide this token to retrieve the subsequent page.
    parent: Required. The parent that owns this collection of session
      templates.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)