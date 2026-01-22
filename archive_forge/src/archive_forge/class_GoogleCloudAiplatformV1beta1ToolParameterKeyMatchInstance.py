from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ToolParameterKeyMatchInstance(_messages.Message):
    """Spec for tool parameter key match instance.

  Fields:
    prediction: Required. Output of the evaluated model.
    reference: Required. Ground truth used to compare against the prediction.
  """
    prediction = _messages.StringField(1)
    reference = _messages.StringField(2)