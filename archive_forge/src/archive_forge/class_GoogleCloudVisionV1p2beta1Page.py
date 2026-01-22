from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p2beta1Page(_messages.Message):
    """Detected page from OCR.

  Fields:
    blocks: List of blocks of text, images etc on this page.
    confidence: Confidence of the OCR results on the page. Range [0, 1].
    height: Page height. For PDFs the unit is points. For images (including
      TIFFs) the unit is pixels.
    property: Additional information detected on the page.
    width: Page width. For PDFs the unit is points. For images (including
      TIFFs) the unit is pixels.
  """
    blocks = _messages.MessageField('GoogleCloudVisionV1p2beta1Block', 1, repeated=True)
    confidence = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    height = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    property = _messages.MessageField('GoogleCloudVisionV1p2beta1TextAnnotationTextProperty', 4)
    width = _messages.IntegerField(5, variant=_messages.Variant.INT32)