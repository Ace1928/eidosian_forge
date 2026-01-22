from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaVideoObjectTrackingAnnotation(_messages.Message):
    """Annotation details specific to video object tracking.

  Fields:
    annotationSpecId: The resource Id of the AnnotationSpec that this
      Annotation pertains to.
    displayName: The display name of the AnnotationSpec that this Annotation
      pertains to.
    instanceId: The instance of the object, expressed as a positive integer.
      Used to track the same object across different frames.
    timeOffset: A time (frame) of a video to which this annotation pertains.
      Represented as the duration since the video's start.
    xMax: The rightmost coordinate of the bounding box.
    xMin: The leftmost coordinate of the bounding box.
    yMax: The bottommost coordinate of the bounding box.
    yMin: The topmost coordinate of the bounding box.
  """
    annotationSpecId = _messages.StringField(1)
    displayName = _messages.StringField(2)
    instanceId = _messages.IntegerField(3)
    timeOffset = _messages.StringField(4)
    xMax = _messages.FloatField(5)
    xMin = _messages.FloatField(6)
    yMax = _messages.FloatField(7)
    yMin = _messages.FloatField(8)