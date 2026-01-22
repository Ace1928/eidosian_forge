from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1QuestionAnsweringCorrectnessInstance(_messages.Message):
    """Spec for question answering correctness instance.

  Fields:
    context: Optional. Text provided as context to answer the question.
    instruction: Required. The question asked and other instruction in the
      inference prompt.
    prediction: Required. Output of the evaluated model.
    reference: Optional. Ground truth used to compare against the prediction.
  """
    context = _messages.StringField(1)
    instruction = _messages.StringField(2)
    prediction = _messages.StringField(3)
    reference = _messages.StringField(4)