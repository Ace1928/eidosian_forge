from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsConfusionMatrixAnnotationSpecRef(_messages.Message):
    """A GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsConfusionMatri
  xAnnotationSpecRef object.

  Fields:
    displayName: Display name of the AnnotationSpec.
    id: ID of the AnnotationSpec.
  """
    displayName = _messages.StringField(1)
    id = _messages.StringField(2)