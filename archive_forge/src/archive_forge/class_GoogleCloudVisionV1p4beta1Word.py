from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p4beta1Word(_messages.Message):
    """A word representation.

  Fields:
    boundingBox: The bounding box for the word. The vertices are in the order
      of top-left, top-right, bottom-right, bottom-left. When a rotation of
      the bounding box is detected the rotation is represented as around the
      top-left corner as defined when the text is read in the 'natural'
      orientation. For example: * when the text is horizontal it might look
      like: 0----1 | | 3----2 * when it's rotated 180 degrees around the top-
      left corner it becomes: 2----3 | | 1----0 and the vertex order will
      still be (0, 1, 2, 3).
    confidence: Confidence of the OCR results for the word. Range [0, 1].
    property: Additional information detected for the word.
    symbols: List of symbols in the word. The order of the symbols follows the
      natural reading order.
  """
    boundingBox = _messages.MessageField('GoogleCloudVisionV1p4beta1BoundingPoly', 1)
    confidence = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    property = _messages.MessageField('GoogleCloudVisionV1p4beta1TextAnnotationTextProperty', 3)
    symbols = _messages.MessageField('GoogleCloudVisionV1p4beta1Symbol', 4, repeated=True)