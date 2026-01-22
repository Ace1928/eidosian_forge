from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1VertexAISearch(_messages.Message):
    """Retrieve from Vertex AI Search datastore for grounding. See
  https://cloud.google.com/vertex-ai-search-and-conversation

  Fields:
    datastore: Required. Fully-qualified Vertex AI Search's datastore resource
      ID. Format: projects/{project}/locations/{location}/collections/{collect
      ion}/dataStores/{dataStore}
  """
    datastore = _messages.StringField(1)