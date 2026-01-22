from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ImportConversationDataRequest(_messages.Message):
    """The request message for ConversationDatasets.ImportConversationData.

  Fields:
    inputConfig: Required. Configuration describing where to import data from.
  """
    inputConfig = _messages.MessageField('GoogleCloudDialogflowV2InputConfig', 1)