from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1beta2PersonDetectionAnnotation(_messages.Message):
    """Person detection annotation per video.

  Fields:
    tracks: The detected tracks of a person.
    version: Feature version.
  """
    tracks = _messages.MessageField('GoogleCloudVideointelligenceV1beta2Track', 1, repeated=True)
    version = _messages.StringField(2)