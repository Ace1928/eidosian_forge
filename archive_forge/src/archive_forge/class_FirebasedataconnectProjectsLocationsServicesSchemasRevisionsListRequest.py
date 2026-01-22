from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirebasedataconnectProjectsLocationsServicesSchemasRevisionsListRequest(_messages.Message):
    """A
  FirebasedataconnectProjectsLocationsServicesSchemasRevisionsListRequest
  object.

  Fields:
    filter: Optional. Filtering results.
    pageSize: Optional. Requested page size. Server may return fewer items
      than requested. If unspecified, server will pick an appropriate default.
    pageToken: Optional. A page token, received from a previous
      `ListSchemaRevisions` call. Provide this to retrieve the subsequent
      page. When paginating, all other parameters provided to
      `ListSchemaRevisions` must match the call that provided the page token.
    parent: Required. Value of parent.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)