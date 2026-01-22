from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListRagCorporaResponse(_messages.Message):
    """Response message for VertexRagDataService.ListRagCorpora.

  Fields:
    nextPageToken: A token to retrieve the next page of results. Pass to
      ListRagCorporaRequest.page_token to obtain that page.
    ragCorpora: List of RagCorpora in the requested page.
  """
    nextPageToken = _messages.StringField(1)
    ragCorpora = _messages.MessageField('GoogleCloudAiplatformV1beta1RagCorpus', 2, repeated=True)