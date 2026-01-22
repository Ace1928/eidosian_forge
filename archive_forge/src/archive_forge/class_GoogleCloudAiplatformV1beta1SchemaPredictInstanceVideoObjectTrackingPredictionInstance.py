from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaPredictInstanceVideoObjectTrackingPredictionInstance(_messages.Message):
    """Prediction input format for Video Object Tracking.

  Fields:
    content: The Google Cloud Storage location of the video on which to
      perform the prediction.
    mimeType: The MIME type of the content of the video. Only the following
      are supported: video/mp4 video/avi video/quicktime
    timeSegmentEnd: The end, exclusive, of the video's time segment on which
      to perform the prediction. Expressed as a number of seconds as measured
      from the start of the video, with "s" appended at the end. Fractions are
      allowed, up to a microsecond precision, and "inf" or "Infinity" is
      allowed, which means the end of the video.
    timeSegmentStart: The beginning, inclusive, of the video's time segment on
      which to perform the prediction. Expressed as a number of seconds as
      measured from the start of the video, with "s" appended at the end.
      Fractions are allowed, up to a microsecond precision.
  """
    content = _messages.StringField(1)
    mimeType = _messages.StringField(2)
    timeSegmentEnd = _messages.StringField(3)
    timeSegmentStart = _messages.StringField(4)