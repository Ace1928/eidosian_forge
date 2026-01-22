from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ExportModelRequest(_messages.Message):
    """Request message for ModelService.ExportModel.

  Fields:
    outputConfig: Required. The desired output location and configuration.
  """
    outputConfig = _messages.MessageField('GoogleCloudAiplatformV1ExportModelRequestOutputConfig', 1)