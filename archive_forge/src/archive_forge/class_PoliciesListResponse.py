from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PoliciesListResponse(_messages.Message):
    """A PoliciesListResponse object.

  Fields:
    header: A ResponseHeader attribute.
    kind: Type of resource.
    nextPageToken: The presence of this field indicates that there exist more
      results following your last page of results in pagination order. To
      fetch them, make another list request using this value as your page
      token. This lets you the complete contents of even very large
      collections one page at a time. However, if the contents of the
      collection change between the first and last paginated list request, the
      set of all elements returned are an inconsistent view of the collection.
      You cannot retrieve a consistent snapshot of a collection larger than
      the maximum page size.
    policies: The policy resources.
  """
    header = _messages.MessageField('ResponseHeader', 1)
    kind = _messages.StringField(2, default='dns#policiesListResponse')
    nextPageToken = _messages.StringField(3)
    policies = _messages.MessageField('Policy', 4, repeated=True)