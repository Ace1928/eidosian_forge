from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1EvaluatedAnnotation(_messages.Message):
    """True positive, false positive, or false negative. EvaluatedAnnotation is
  only available under ModelEvaluationSlice with slice of `annotationSpec`
  dimension.

  Enums:
    TypeValueValuesEnum: Output only. Type of the EvaluatedAnnotation.

  Fields:
    dataItemPayload: Output only. The data item payload that the Model
      predicted this EvaluatedAnnotation on.
    errorAnalysisAnnotations: Annotations of model error analysis results.
    evaluatedDataItemViewId: Output only. ID of the EvaluatedDataItemView
      under the same ancestor ModelEvaluation. The EvaluatedDataItemView
      consists of all ground truths and predictions on data_item_payload.
    explanations: Explanations of predictions. Each element of the
      explanations indicates the explanation for one explanation Method. The
      attributions list in the EvaluatedAnnotationExplanation.explanation
      object corresponds to the predictions list. For example, the second
      element in the attributions list explains the second element in the
      predictions list.
    groundTruths: Output only. The ground truth Annotations, i.e. the
      Annotations that exist in the test data the Model is evaluated on. For
      true positive, there is one and only one ground truth annotation, which
      matches the only prediction in predictions. For false positive, there
      are zero or more ground truth annotations that are similar to the only
      prediction in predictions, but not enough for a match. For false
      negative, there is one and only one ground truth annotation, which
      doesn't match any predictions created by the model. The schema of the
      ground truth is stored in ModelEvaluation.annotation_schema_uri
    predictions: Output only. The model predicted annotations. For true
      positive, there is one and only one prediction, which matches the only
      one ground truth annotation in ground_truths. For false positive, there
      is one and only one prediction, which doesn't match any ground truth
      annotation of the corresponding data_item_view_id. For false negative,
      there are zero or more predictions which are similar to the only ground
      truth annotation in ground_truths but not enough for a match. The schema
      of the prediction is stored in ModelEvaluation.annotation_schema_uri
    type: Output only. Type of the EvaluatedAnnotation.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Output only. Type of the EvaluatedAnnotation.

    Values:
      EVALUATED_ANNOTATION_TYPE_UNSPECIFIED: Invalid value.
      TRUE_POSITIVE: The EvaluatedAnnotation is a true positive. It has a
        prediction created by the Model and a ground truth Annotation which
        the prediction matches.
      FALSE_POSITIVE: The EvaluatedAnnotation is false positive. It has a
        prediction created by the Model which does not match any ground truth
        annotation.
      FALSE_NEGATIVE: The EvaluatedAnnotation is false negative. It has a
        ground truth annotation which is not matched by any of the model
        created predictions.
    """
        EVALUATED_ANNOTATION_TYPE_UNSPECIFIED = 0
        TRUE_POSITIVE = 1
        FALSE_POSITIVE = 2
        FALSE_NEGATIVE = 3
    dataItemPayload = _messages.MessageField('extra_types.JsonValue', 1)
    errorAnalysisAnnotations = _messages.MessageField('GoogleCloudAiplatformV1beta1ErrorAnalysisAnnotation', 2, repeated=True)
    evaluatedDataItemViewId = _messages.StringField(3)
    explanations = _messages.MessageField('GoogleCloudAiplatformV1beta1EvaluatedAnnotationExplanation', 4, repeated=True)
    groundTruths = _messages.MessageField('extra_types.JsonValue', 5, repeated=True)
    predictions = _messages.MessageField('extra_types.JsonValue', 6, repeated=True)
    type = _messages.EnumField('TypeValueValuesEnum', 7)