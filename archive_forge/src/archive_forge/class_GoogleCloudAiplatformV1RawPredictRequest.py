from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1RawPredictRequest(_messages.Message):
    """Request message for PredictionService.RawPredict.

  Fields:
    httpBody: The prediction input. Supports HTTP headers and arbitrary data
      payload. A DeployedModel may have an upper limit on the number of
      instances it supports per request. When this limit it is exceeded for an
      AutoML model, the RawPredict method returns an error. When this limit is
      exceeded for a custom-trained model, the behavior varies depending on
      the model. You can specify the schema for each instance in the
      predict_schemata.instance_schema_uri field when you create a Model. This
      schema applies when you deploy the `Model` as a `DeployedModel` to an
      Endpoint and use the `RawPredict` method.
  """
    httpBody = _messages.MessageField('GoogleApiHttpBody', 1)