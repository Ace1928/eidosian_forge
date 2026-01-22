from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClassificationCategory(_messages.Message):
    """Represents a category returned from the text classifier.

  Fields:
    confidence: The classifier's confidence of the category. Number represents
      how certain the classifier is that this category represents the given
      text.
    name: The name of the category representing the document.
  """
    confidence = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    name = _messages.StringField(2)