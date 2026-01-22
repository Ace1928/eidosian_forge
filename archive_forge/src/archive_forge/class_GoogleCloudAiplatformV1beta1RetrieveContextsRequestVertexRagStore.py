from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1RetrieveContextsRequestVertexRagStore(_messages.Message):
    """The data source for Vertex RagStore.

  Fields:
    ragCorpora: Required. RagCorpora resource name. Format:
      `projects/{project}/locations/{location}/ragCorpora/{rag_corpus}`
      Currently only one corpus is allowed. In the future we may open up
      multiple corpora support. However, they should be from the same project
      and location.
  """
    ragCorpora = _messages.StringField(1, repeated=True)