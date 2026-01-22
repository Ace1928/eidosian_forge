from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudnumberregistryProjectsLocationsRegistryBooksSearchRegistryRequest(_messages.Message):
    """A CloudnumberregistryProjectsLocationsRegistryBooksSearchRegistryRequest
  object.

  Fields:
    attributeKeys: Optional. A list of attribute keys owned by the registry
      nodes.
    book: Required. Name of the resource
    ipRange: Optional. IP range to filter for registry node.
    keywords: Optional. A list of keywords that are contained by the attribute
      values within registry nodes.
    orderBy: Optional. Hint for how to order the results
    pageSize: Optional. Requested page size. Server may return fewer items
      than requested. If unspecified, server will pick an appropriate default.
    pageToken: Optional. A token identifying a page of results the server
      should return.
    source: Optional. Source filter of the registry nodes.
  """
    attributeKeys = _messages.StringField(1, repeated=True)
    book = _messages.StringField(2, required=True)
    ipRange = _messages.StringField(3)
    keywords = _messages.StringField(4, repeated=True)
    orderBy = _messages.StringField(5)
    pageSize = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(7)
    source = _messages.StringField(8)