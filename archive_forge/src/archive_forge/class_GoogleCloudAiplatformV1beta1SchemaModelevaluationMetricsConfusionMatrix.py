from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsConfusionMatrix(_messages.Message):
    """A
  GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsConfusionMatrix
  object.

  Messages:
    RowsValueListEntry: Single entry in a RowsValue.

  Fields:
    annotationSpecs: AnnotationSpecs used in the confusion matrix. For AutoML
      Text Extraction, a special negative AnnotationSpec with empty `id` and
      `displayName` of "NULL" will be added as the last element.
    rows: Rows in the confusion matrix. The number of rows is equal to the
      size of `annotationSpecs`. `rowsi` is the number of DataItems that have
      ground truth of the `annotationSpecs[i]` and are predicted as
      `annotationSpecs[j]` by the Model being evaluated. For Text Extraction,
      when `annotationSpecs[i]` is the last element in `annotationSpecs`, i.e.
      the special negative AnnotationSpec, `rowsi` is the number of predicted
      entities of `annoatationSpec[j]` that are not labeled as any of the
      ground truth AnnotationSpec. When annotationSpecs[j] is the special
      negative AnnotationSpec, `rowsi` is the number of entities have ground
      truth of `annotationSpec[i]` that are not predicted as an entity by the
      Model. The value of the last cell, i.e. `rowi` where i == j and
      `annotationSpec[i]` is the special negative AnnotationSpec, is always 0.
  """

    class RowsValueListEntry(_messages.Message):
        """Single entry in a RowsValue.

    Fields:
      entry: A extra_types.JsonValue attribute.
    """
        entry = _messages.MessageField('extra_types.JsonValue', 1, repeated=True)
    annotationSpecs = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsConfusionMatrixAnnotationSpecRef', 1, repeated=True)
    rows = _messages.MessageField('RowsValueListEntry', 2, repeated=True)