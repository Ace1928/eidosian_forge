from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1RagContexts(_messages.Message):
    """Relevant contexts for one query.

  Fields:
    contexts: All its contexts.
  """
    contexts = _messages.MessageField('GoogleCloudAiplatformV1beta1RagContextsContext', 1, repeated=True)