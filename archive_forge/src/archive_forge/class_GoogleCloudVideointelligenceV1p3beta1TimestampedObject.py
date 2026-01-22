from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p3beta1TimestampedObject(_messages.Message):
    """For tracking related features. An object at time_offset with attributes,
  and located with normalized_bounding_box.

  Fields:
    attributes: Optional. The attributes of the object in the bounding box.
    landmarks: Optional. The detected landmarks.
    normalizedBoundingBox: Normalized Bounding box in a frame, where the
      object is located.
    timeOffset: Time-offset, relative to the beginning of the video,
      corresponding to the video frame for this object.
  """
    attributes = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1DetectedAttribute', 1, repeated=True)
    landmarks = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1DetectedLandmark', 2, repeated=True)
    normalizedBoundingBox = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1NormalizedBoundingBox', 3)
    timeOffset = _messages.StringField(4)