from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaPredictPredictionClassificationPredictionResult(_messages.Message):
    """Prediction output format for Image and Text Classification.

  Fields:
    confidences: The Model's confidences in correctness of the predicted IDs,
      higher value means higher confidence. Order matches the Ids.
    displayNames: The display names of the AnnotationSpecs that had been
      identified, order matches the IDs.
    ids: The resource IDs of the AnnotationSpecs that had been identified.
  """
    confidences = _messages.FloatField(1, repeated=True, variant=_messages.Variant.FLOAT)
    displayNames = _messages.StringField(2, repeated=True)
    ids = _messages.IntegerField(3, repeated=True)