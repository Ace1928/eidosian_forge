from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ListVersionsResponse(_messages.Message):
    """The response message for Versions.ListVersions.

  Fields:
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
    versions: The list of agent versions. There will be a maximum number of
      items returned based on the page_size field in the request.
  """
    nextPageToken = _messages.StringField(1)
    versions = _messages.MessageField('GoogleCloudDialogflowV2Version', 2, repeated=True)