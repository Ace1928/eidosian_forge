from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PredictRequest(_messages.Message):
    """Request message for PredictionService.Predict.

  Fields:
    instances: Required. The instances that are the input to the prediction
      call. A DeployedModel may have an upper limit on the number of instances
      it supports per request, and when it is exceeded the prediction call
      errors in case of AutoML Models, or, in case of customer created Models,
      the behaviour is as documented by that Model. The schema of any single
      instance may be specified via Endpoint's DeployedModels' Model's
      PredictSchemata's instance_schema_uri.
    parameters: The parameters that govern the prediction. The schema of the
      parameters may be specified via Endpoint's DeployedModels' Model's
      PredictSchemata's parameters_schema_uri.
  """
    instances = _messages.MessageField('extra_types.JsonValue', 1, repeated=True)
    parameters = _messages.MessageField('extra_types.JsonValue', 2)