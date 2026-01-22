from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ExportDataRequest(_messages.Message):
    """Request message for DatasetService.ExportData.

  Fields:
    exportConfig: Required. The desired output location.
  """
    exportConfig = _messages.MessageField('GoogleCloudAiplatformV1ExportDataConfig', 1)