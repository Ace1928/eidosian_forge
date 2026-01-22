from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1QuestionAnsweringQualityInstance(_messages.Message):
    """Spec for question answering quality instance.

  Fields:
    context: Required. Text to answer the question.
    instruction: Required. Question Answering prompt for LLM.
    prediction: Required. Output of the evaluated model.
    reference: Optional. Ground truth used to compare against the prediction.
  """
    context = _messages.StringField(1)
    instruction = _messages.StringField(2)
    prediction = _messages.StringField(3)
    reference = _messages.StringField(4)