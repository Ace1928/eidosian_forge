from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1ObjectTrackingFrame(_messages.Message):
    """Video frame level annotations for object detection and tracking. This
  field stores per frame location, time offset, and confidence.

  Fields:
    normalizedBoundingBox: The normalized bounding box location of this object
      track for the frame.
    timeOffset: The timestamp of the frame in microseconds.
  """
    normalizedBoundingBox = _messages.MessageField('GoogleCloudVideointelligenceV1NormalizedBoundingBox', 1)
    timeOffset = _messages.StringField(2)