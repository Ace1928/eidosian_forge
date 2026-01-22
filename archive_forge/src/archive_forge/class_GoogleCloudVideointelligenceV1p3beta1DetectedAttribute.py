from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p3beta1DetectedAttribute(_messages.Message):
    """A generic detected attribute represented by name in string format.

  Fields:
    confidence: Detected attribute confidence. Range [0, 1].
    name: The name of the attribute, for example, glasses, dark_glasses,
      mouth_open. A full list of supported type names will be provided in the
      document.
    value: Text value of the detection result. For example, the value for
      "HairColor" can be "black", "blonde", etc.
  """
    confidence = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    name = _messages.StringField(2)
    value = _messages.StringField(3)