from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerflexProjectsLocationsContextsListRequest(_messages.Message):
    """A ComposerflexProjectsLocationsContextsListRequest object.

  Fields:
    filter: Filter will remain internal until its future implementation.
    orderBy: Optional. Specifies the ordering of results following syntax at
      https://cloud.google.com/apis/design/design_patterns#sorting_order.
      Order by will remain internal until its future implementation.
    pageSize: The maximum number of contexts to return.
    pageToken: Optional. The next_page_token returned from a previous List
      request.
    parent: List contexts in the given parent resource. Parent must be in the
      form "projects/{projectId}/locations/{locationId}".
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)