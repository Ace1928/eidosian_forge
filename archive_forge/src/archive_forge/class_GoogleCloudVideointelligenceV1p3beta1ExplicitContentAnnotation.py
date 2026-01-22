from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p3beta1ExplicitContentAnnotation(_messages.Message):
    """Explicit content annotation (based on per-frame visual signals only). If
  no explicit content has been detected in a frame, no annotations are present
  for that frame.

  Fields:
    frames: All video frames where explicit content was detected.
    version: Feature version.
  """
    frames = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1ExplicitContentFrame', 1, repeated=True)
    version = _messages.StringField(2)