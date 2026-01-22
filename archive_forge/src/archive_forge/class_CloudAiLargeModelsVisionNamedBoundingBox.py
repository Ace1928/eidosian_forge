from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiLargeModelsVisionNamedBoundingBox(_messages.Message):
    """A CloudAiLargeModelsVisionNamedBoundingBox object.

  Fields:
    classes: A string attribute.
    entities: A string attribute.
    scores: A number attribute.
    x1: A number attribute.
    x2: A number attribute.
    y1: A number attribute.
    y2: A number attribute.
  """
    classes = _messages.StringField(1, repeated=True)
    entities = _messages.StringField(2, repeated=True)
    scores = _messages.FloatField(3, repeated=True, variant=_messages.Variant.FLOAT)
    x1 = _messages.FloatField(4, variant=_messages.Variant.FLOAT)
    x2 = _messages.FloatField(5, variant=_messages.Variant.FLOAT)
    y1 = _messages.FloatField(6, variant=_messages.Variant.FLOAT)
    y2 = _messages.FloatField(7, variant=_messages.Variant.FLOAT)