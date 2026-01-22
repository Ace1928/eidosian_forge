from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
class ResourceRecordSetsListResponse(_messages.Message):
    """A ResourceRecordSetsListResponse object.

  Fields:
    kind: Type of resource.
    nextPageToken: The presence of this field indicates that there exist more
      results following your last page of results in pagination order. To
      fetch them, make another list request using this value as your
      pagination token.  In this way you can retrieve the complete contents of
      even very large collections one page at a time. However, if the contents
      of the collection change between the first and last paginated list
      request, the set of all elements returned will be an inconsistent view
      of the collection. There is no way to retrieve a consistent snapshot of
      a collection larger than the maximum page size.
    rrsets: The resource record set resources.
  """
    kind = _messages.StringField(1, default=u'dns#resourceRecordSetsListResponse')
    nextPageToken = _messages.StringField(2)
    rrsets = _messages.MessageField('ResourceRecordSet', 3, repeated=True)