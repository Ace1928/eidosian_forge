from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaPredictPredictionVideoObjectTrackingPredictionResultFrame(_messages.Message):
    """The fields `xMin`, `xMax`, `yMin`, and `yMax` refer to a bounding box,
  i.e. the rectangle over the video frame pinpointing the found
  AnnotationSpec. The coordinates are relative to the frame size, and the
  point 0,0 is in the top left of the frame.

  Fields:
    timeOffset: A time (frame) of a video in which the object has been
      detected. Expressed as a number of seconds as measured from the start of
      the video, with fractions up to a microsecond precision, and with "s"
      appended at the end.
    xMax: The rightmost coordinate of the bounding box.
    xMin: The leftmost coordinate of the bounding box.
    yMax: The bottommost coordinate of the bounding box.
    yMin: The topmost coordinate of the bounding box.
  """
    timeOffset = _messages.StringField(1)
    xMax = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    xMin = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    yMax = _messages.FloatField(4, variant=_messages.Variant.FLOAT)
    yMin = _messages.FloatField(5, variant=_messages.Variant.FLOAT)