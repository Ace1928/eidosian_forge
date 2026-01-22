from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ImportModelEvaluationRequest(_messages.Message):
    """Request message for ModelService.ImportModelEvaluation

  Fields:
    modelEvaluation: Required. Model evaluation resource to be imported.
  """
    modelEvaluation = _messages.MessageField('GoogleCloudAiplatformV1beta1ModelEvaluation', 1)