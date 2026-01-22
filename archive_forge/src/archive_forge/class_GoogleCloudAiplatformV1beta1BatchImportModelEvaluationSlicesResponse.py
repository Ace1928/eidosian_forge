from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1BatchImportModelEvaluationSlicesResponse(_messages.Message):
    """Response message for ModelService.BatchImportModelEvaluationSlices

  Fields:
    importedModelEvaluationSlices: Output only. List of imported
      ModelEvaluationSlice.name.
  """
    importedModelEvaluationSlices = _messages.StringField(1, repeated=True)