from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListRagFilesResponse(_messages.Message):
    """Response message for VertexRagDataService.ListRagFiles.

  Fields:
    nextPageToken: A token to retrieve the next page of results. Pass to
      ListRagFilesRequest.page_token to obtain that page.
    ragFiles: List of RagFiles in the requested page.
  """
    nextPageToken = _messages.StringField(1)
    ragFiles = _messages.MessageField('GoogleCloudAiplatformV1beta1RagFile', 2, repeated=True)