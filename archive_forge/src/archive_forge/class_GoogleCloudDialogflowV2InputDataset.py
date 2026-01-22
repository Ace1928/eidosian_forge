from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2InputDataset(_messages.Message):
    """InputDataset used to create model or do evaluation. NextID:5

  Fields:
    dataset: Required. ConversationDataset resource name. Format:
      `projects//locations//conversationDatasets/`
  """
    dataset = _messages.StringField(1)