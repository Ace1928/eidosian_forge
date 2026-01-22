from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsAnswerRecordsListRequest(_messages.Message):
    """A DialogflowProjectsLocationsAnswerRecordsListRequest object.

  Fields:
    filter: Optional. Filters to restrict results to specific answer records.
      Marked deprecated as it hasn't been, and isn't currently, supported. For
      more information about filtering, see [API
      Filtering](https://aip.dev/160).
    pageSize: Optional. The maximum number of records to return in a single
      page. The server may return fewer records than this. If unspecified, we
      use 10. The maximum is 100.
    pageToken: Optional. The ListAnswerRecordsResponse.next_page_token value
      returned from a previous list request used to continue listing on the
      next page.
    parent: Required. The project to list all answer records for in reverse
      chronological order. Format: `projects//locations/`.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)