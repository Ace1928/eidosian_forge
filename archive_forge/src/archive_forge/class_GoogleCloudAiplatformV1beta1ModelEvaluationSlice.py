from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ModelEvaluationSlice(_messages.Message):
    """A collection of metrics calculated by comparing Model's predictions on a
  slice of the test data against ground truth annotations.

  Fields:
    createTime: Output only. Timestamp when this ModelEvaluationSlice was
      created.
    metrics: Output only. Sliced evaluation metrics of the Model. The schema
      of the metrics is stored in metrics_schema_uri
    metricsSchemaUri: Output only. Points to a YAML file stored on Google
      Cloud Storage describing the metrics of this ModelEvaluationSlice. The
      schema is defined as an OpenAPI 3.0.2 [Schema
      Object](https://github.com/OAI/OpenAPI-
      Specification/blob/main/versions/3.0.2.md#schemaObject).
    modelExplanation: Output only. Aggregated explanation metrics for the
      Model's prediction output over the data this ModelEvaluation uses. This
      field is populated only if the Model is evaluated with explanations, and
      only for tabular Models.
    name: Output only. The resource name of the ModelEvaluationSlice.
    slice: Output only. The slice of the test data that is used to evaluate
      the Model.
  """
    createTime = _messages.StringField(1)
    metrics = _messages.MessageField('extra_types.JsonValue', 2)
    metricsSchemaUri = _messages.StringField(3)
    modelExplanation = _messages.MessageField('GoogleCloudAiplatformV1beta1ModelExplanation', 4)
    name = _messages.StringField(5)
    slice = _messages.MessageField('GoogleCloudAiplatformV1beta1ModelEvaluationSliceSlice', 6)