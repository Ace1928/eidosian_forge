from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p3beta1CelebrityRecognitionAnnotation(_messages.Message):
    """Celebrity recognition annotation per video.

  Fields:
    celebrityTracks: The tracks detected from the input video, including
      recognized celebrities and other detected faces in the video.
    version: Feature version.
  """
    celebrityTracks = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1CelebrityTrack', 1, repeated=True)
    version = _messages.StringField(2)