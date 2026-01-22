from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p3beta1CelebrityTrack(_messages.Message):
    """The annotation result of a celebrity face track. RecognizedCelebrity
  field could be empty if the face track does not have any matched
  celebrities.

  Fields:
    celebrities: Top N match of the celebrities for the face in this track.
    faceTrack: A track of a person's face.
  """
    celebrities = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1RecognizedCelebrity', 1, repeated=True)
    faceTrack = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1Track', 2)