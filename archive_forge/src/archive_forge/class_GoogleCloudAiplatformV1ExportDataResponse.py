from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ExportDataResponse(_messages.Message):
    """Response message for DatasetService.ExportData.

  Fields:
    dataStats: Only present for custom code training export use case. Records
      data stats, i.e., train/validation/test item/annotation counts
      calculated during the export operation.
    exportedFiles: All of the files that are exported in this export
      operation. For custom code training export, only three (training,
      validation and test) Cloud Storage paths in wildcard format are
      populated (for example, gs://.../training-*).
  """
    dataStats = _messages.MessageField('GoogleCloudAiplatformV1ModelDataStats', 1)
    exportedFiles = _messages.StringField(2, repeated=True)