from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2CreateConversationDatasetOperationMetadata(_messages.Message):
    """Metadata for ConversationDatasets.

  Fields:
    conversationDataset: The resource name of the conversation dataset that
      will be created. Format: `projects//locations//conversationDatasets/`
  """
    conversationDataset = _messages.StringField(1)