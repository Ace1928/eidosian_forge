from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1PredictResponse(_messages.Message):
    """Response message for PredictionService.Predict.

  Fields:
    deployedModelId: ID of the Endpoint's DeployedModel that served this
      prediction.
    metadata: Output only. Request-level metadata returned by the model. The
      metadata type will be dependent upon the model implementation.
    model: Output only. The resource name of the Model which is deployed as
      the DeployedModel that this prediction hits.
    modelDisplayName: Output only. The display name of the Model which is
      deployed as the DeployedModel that this prediction hits.
    modelVersionId: Output only. The version ID of the Model which is deployed
      as the DeployedModel that this prediction hits.
    predictions: The predictions that are the output of the predictions call.
      The schema of any single prediction may be specified via Endpoint's
      DeployedModels' Model's PredictSchemata's prediction_schema_uri.
  """
    deployedModelId = _messages.StringField(1)
    metadata = _messages.MessageField('extra_types.JsonValue', 2)
    model = _messages.StringField(3)
    modelDisplayName = _messages.StringField(4)
    modelVersionId = _messages.StringField(5)
    predictions = _messages.MessageField('extra_types.JsonValue', 6, repeated=True)