from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1ListPolicyTagsResponse(_messages.Message):
    """Response message for ListPolicyTags.

  Fields:
    nextPageToken: Pagination token of the next results page. Empty if there
      are no more results in the list.
    policyTags: The policy tags that belong to the taxonomy.
  """
    nextPageToken = _messages.StringField(1)
    policyTags = _messages.MessageField('GoogleCloudDatacatalogV1PolicyTag', 2, repeated=True)