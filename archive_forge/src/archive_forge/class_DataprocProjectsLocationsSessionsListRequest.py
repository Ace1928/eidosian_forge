from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsSessionsListRequest(_messages.Message):
    """A DataprocProjectsLocationsSessionsListRequest object.

  Fields:
    filter: Optional. A filter for the sessions to return in the response.A
      filter is a logical expression constraining the values of various fields
      in each session resource. Filters are case sensitive, and may contain
      multiple clauses combined with logical operators (AND, OR). Supported
      fields are session_id, session_uuid, state, create_time, and
      labels.Example: state = ACTIVE and create_time < "2023-01-01T00:00:00Z"
      is a filter for sessions in an ACTIVE state that were created before
      2023-01-01. state = ACTIVE and labels.environment=production is a filter
      for sessions in an ACTIVE state that have a production environment
      label.See https://google.aip.dev/assets/misc/ebnf-filtering.txt for a
      detailed description of the filter syntax and a list of supported
      comparators.
    orderBy: Optional. Field(s) on which to sort the list of sessions. See
      https://google.aip.dev/132#ordering for more information.
    pageSize: Optional. The maximum number of sessions to return in each
      response. The service may return fewer than this value.
    pageToken: Optional. A page token received from a previous ListSessions
      call. Provide this token to retrieve the subsequent page.
    parent: Required. The parent, which owns this collection of sessions.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)