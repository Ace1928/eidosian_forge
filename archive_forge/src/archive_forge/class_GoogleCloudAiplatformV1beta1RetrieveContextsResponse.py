from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1RetrieveContextsResponse(_messages.Message):
    """Response message for VertexRagService.RetrieveContexts.

  Fields:
    contexts: The contexts of the query.
  """
    contexts = _messages.MessageField('GoogleCloudAiplatformV1beta1RagContexts', 1)