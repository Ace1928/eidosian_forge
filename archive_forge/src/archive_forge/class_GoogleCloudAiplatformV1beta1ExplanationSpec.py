from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ExplanationSpec(_messages.Message):
    """Specification of Model explanation.

  Fields:
    metadata: Optional. Metadata describing the Model's input and output for
      explanation.
    parameters: Required. Parameters that configure explaining of the Model's
      predictions.
  """
    metadata = _messages.MessageField('GoogleCloudAiplatformV1beta1ExplanationMetadata', 1)
    parameters = _messages.MessageField('GoogleCloudAiplatformV1beta1ExplanationParameters', 2)