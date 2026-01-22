from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1FaceFrame(_messages.Message):
    """Deprecated. No effect.

  Fields:
    normalizedBoundingBoxes: Normalized Bounding boxes in a frame. There can
      be more than one boxes if the same face is detected in multiple
      locations within the current frame.
    timeOffset: Time-offset, relative to the beginning of the video,
      corresponding to the video frame for this location.
  """
    normalizedBoundingBoxes = _messages.MessageField('GoogleCloudVideointelligenceV1NormalizedBoundingBox', 1, repeated=True)
    timeOffset = _messages.StringField(2)