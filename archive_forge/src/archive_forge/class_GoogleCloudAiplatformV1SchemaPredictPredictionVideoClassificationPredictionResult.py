from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaPredictPredictionVideoClassificationPredictionResult(_messages.Message):
    """Prediction output format for Video Classification.

  Fields:
    confidence: The Model's confidence in correction of this prediction,
      higher value means higher confidence.
    displayName: The display name of the AnnotationSpec that had been
      identified.
    id: The resource ID of the AnnotationSpec that had been identified.
    timeSegmentEnd: The end, exclusive, of the video's time segment in which
      the AnnotationSpec has been identified. Expressed as a number of seconds
      as measured from the start of the video, with fractions up to a
      microsecond precision, and with "s" appended at the end. Note that for
      'segment-classification' prediction type, this equals the original
      'timeSegmentEnd' from the input instance, for other types it is the end
      of a shot or a 1 second interval respectively.
    timeSegmentStart: The beginning, inclusive, of the video's time segment in
      which the AnnotationSpec has been identified. Expressed as a number of
      seconds as measured from the start of the video, with fractions up to a
      microsecond precision, and with "s" appended at the end. Note that for
      'segment-classification' prediction type, this equals the original
      'timeSegmentStart' from the input instance, for other types it is the
      start of a shot or a 1 second interval respectively.
    type: The type of the prediction. The requested types can be configured
      via parameters. This will be one of - segment-classification - shot-
      classification - one-sec-interval-classification
  """
    confidence = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    displayName = _messages.StringField(2)
    id = _messages.StringField(3)
    timeSegmentEnd = _messages.StringField(4)
    timeSegmentStart = _messages.StringField(5)
    type = _messages.StringField(6)