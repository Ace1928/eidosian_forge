from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsIndexesListRequest(_messages.Message):
    """A AiplatformProjectsLocationsIndexesListRequest object.

  Fields:
    filter: The standard list filter.
    pageSize: The standard list page size.
    pageToken: The standard list page token. Typically obtained via
      ListIndexesResponse.next_page_token of the previous
      IndexService.ListIndexes call.
    parent: Required. The resource name of the Location from which to list the
      Indexes. Format: `projects/{project}/locations/{location}`
    readMask: Mask specifying which fields to read.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
    readMask = _messages.StringField(5)