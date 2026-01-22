from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1RuntimeConfigVertexAISearchRuntimeConfig(_messages.Message):
    """A GoogleCloudAiplatformV1beta1RuntimeConfigVertexAISearchRuntimeConfig
  object.

  Fields:
    servingConfigName: Required. Vertext AI Search serving config name.
      Format: `projects/{project}/locations/{location}/collections/{collection
      }/engines/{engine}/servingConfigs/{serving_config}` or `projects/{projec
      t}/locations/{location}/collections/{collection}/dataStores/{data_store}
      /servingConfigs/{serving_config}`
  """
    servingConfigName = _messages.StringField(1)