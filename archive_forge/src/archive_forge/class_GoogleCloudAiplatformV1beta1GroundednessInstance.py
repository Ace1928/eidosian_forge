from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1GroundednessInstance(_messages.Message):
    """Spec for groundedness instance.

  Fields:
    context: Required. Background information provided in context used to
      compare against the prediction.
    prediction: Required. Output of the evaluated model.
  """
    context = _messages.StringField(1)
    prediction = _messages.StringField(2)