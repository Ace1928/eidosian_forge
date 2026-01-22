from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ExportOperationMetadata(_messages.Message):
    """Metadata related to the Export Data Operations (e.g. ExportDocument).

  Fields:
    exportedGcsDestination: Cloud Storage file path of the exported data.
  """
    exportedGcsDestination = _messages.MessageField('GoogleCloudDialogflowV2GcsDestination', 1)