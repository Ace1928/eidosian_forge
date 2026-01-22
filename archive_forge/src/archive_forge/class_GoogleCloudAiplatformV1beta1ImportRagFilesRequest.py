from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ImportRagFilesRequest(_messages.Message):
    """Request message for VertexRagDataService.ImportRagFiles.

  Fields:
    importRagFilesConfig: Required. The config for the RagFiles to be synced
      and imported into the RagCorpus. VertexRagDataService.ImportRagFiles.
  """
    importRagFilesConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1ImportRagFilesConfig', 1)