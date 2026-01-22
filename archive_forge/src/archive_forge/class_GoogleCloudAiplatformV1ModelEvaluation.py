from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ModelEvaluation(_messages.Message):
    """A collection of metrics calculated by comparing Model's predictions on
  all of the test data against annotations from the test data.

  Fields:
    annotationSchemaUri: Points to a YAML file stored on Google Cloud Storage
      describing EvaluatedDataItemView.predictions,
      EvaluatedDataItemView.ground_truths, EvaluatedAnnotation.predictions,
      and EvaluatedAnnotation.ground_truths. The schema is defined as an
      OpenAPI 3.0.2 [Schema Object](https://github.com/OAI/OpenAPI-
      Specification/blob/main/versions/3.0.2.md#schemaObject). This field is
      not populated if there are neither EvaluatedDataItemViews nor
      EvaluatedAnnotations under this ModelEvaluation.
    createTime: Output only. Timestamp when this ModelEvaluation was created.
    dataItemSchemaUri: Points to a YAML file stored on Google Cloud Storage
      describing EvaluatedDataItemView.data_item_payload and
      EvaluatedAnnotation.data_item_payload. The schema is defined as an
      OpenAPI 3.0.2 [Schema Object](https://github.com/OAI/OpenAPI-
      Specification/blob/main/versions/3.0.2.md#schemaObject). This field is
      not populated if there are neither EvaluatedDataItemViews nor
      EvaluatedAnnotations under this ModelEvaluation.
    displayName: The display name of the ModelEvaluation.
    explanationSpecs: Describes the values of ExplanationSpec that are used
      for explaining the predicted values on the evaluated data.
    metadata: The metadata of the ModelEvaluation. For the ModelEvaluation
      uploaded from Managed Pipeline, metadata contains a structured value
      with keys of "pipeline_job_id", "evaluation_dataset_type",
      "evaluation_dataset_path", "row_based_metrics_path".
    metrics: Evaluation metrics of the Model. The schema of the metrics is
      stored in metrics_schema_uri
    metricsSchemaUri: Points to a YAML file stored on Google Cloud Storage
      describing the metrics of this ModelEvaluation. The schema is defined as
      an OpenAPI 3.0.2 [Schema Object](https://github.com/OAI/OpenAPI-
      Specification/blob/main/versions/3.0.2.md#schemaObject).
    modelExplanation: Aggregated explanation metrics for the Model's
      prediction output over the data this ModelEvaluation uses. This field is
      populated only if the Model is evaluated with explanations, and only for
      AutoML tabular Models.
    name: Output only. The resource name of the ModelEvaluation.
    sliceDimensions: All possible dimensions of ModelEvaluationSlices. The
      dimensions can be used as the filter of the
      ModelService.ListModelEvaluationSlices request, in the form of
      `slice.dimension = `.
  """
    annotationSchemaUri = _messages.StringField(1)
    createTime = _messages.StringField(2)
    dataItemSchemaUri = _messages.StringField(3)
    displayName = _messages.StringField(4)
    explanationSpecs = _messages.MessageField('GoogleCloudAiplatformV1ModelEvaluationModelEvaluationExplanationSpec', 5, repeated=True)
    metadata = _messages.MessageField('extra_types.JsonValue', 6)
    metrics = _messages.MessageField('extra_types.JsonValue', 7)
    metricsSchemaUri = _messages.StringField(8)
    modelExplanation = _messages.MessageField('GoogleCloudAiplatformV1ModelExplanation', 9)
    name = _messages.StringField(10)
    sliceDimensions = _messages.StringField(11, repeated=True)