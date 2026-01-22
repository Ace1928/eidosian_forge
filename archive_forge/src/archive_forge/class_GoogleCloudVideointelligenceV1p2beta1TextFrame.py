from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p2beta1TextFrame(_messages.Message):
    """Video frame level annotation results for text annotation (OCR). Contains
  information regarding timestamp and bounding box locations for the frames
  containing detected OCR text snippets.

  Fields:
    rotatedBoundingBox: Bounding polygon of the detected text for this frame.
    timeOffset: Timestamp of this frame.
  """
    rotatedBoundingBox = _messages.MessageField('GoogleCloudVideointelligenceV1p2beta1NormalizedBoundingPoly', 1)
    timeOffset = _messages.StringField(2)