from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ModelEvaluationSliceSliceSliceSpecValue(_messages.Message):
    """Single value that supports strings and floats.

  Fields:
    floatValue: Float type.
    stringValue: String type.
  """
    floatValue = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    stringValue = _messages.StringField(2)