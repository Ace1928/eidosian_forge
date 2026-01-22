from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ImportConversationDataOperationResponse(_messages.Message):
    """Response used for ConversationDatasets.ImportConversationData long
  running operation.

  Fields:
    conversationDataset: The resource name of the imported conversation
      dataset. Format: `projects//locations//conversationDatasets/`
    importCount: Number of conversations imported successfully.
  """
    conversationDataset = _messages.StringField(1)
    importCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)