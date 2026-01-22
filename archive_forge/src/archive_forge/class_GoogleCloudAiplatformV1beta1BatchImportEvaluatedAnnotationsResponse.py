from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1BatchImportEvaluatedAnnotationsResponse(_messages.Message):
    """Response message for ModelService.BatchImportEvaluatedAnnotations

  Fields:
    importedEvaluatedAnnotationsCount: Output only. Number of
      EvaluatedAnnotations imported.
  """
    importedEvaluatedAnnotationsCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)