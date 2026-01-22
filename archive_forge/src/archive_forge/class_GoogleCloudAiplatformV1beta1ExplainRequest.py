from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ExplainRequest(_messages.Message):
    """Request message for PredictionService.Explain.

  Messages:
    ConcurrentExplanationSpecOverrideValue: Optional. This field is the same
      as the one above, but supports multiple explanations to occur in
      parallel. The key can be any string. Each override will be run against
      the model, then its explanations will be grouped together. Note - these
      explanations are run **In Addition** to the default Explanation in the
      deployed model.

  Fields:
    concurrentExplanationSpecOverride: Optional. This field is the same as the
      one above, but supports multiple explanations to occur in parallel. The
      key can be any string. Each override will be run against the model, then
      its explanations will be grouped together. Note - these explanations are
      run **In Addition** to the default Explanation in the deployed model.
    deployedModelId: If specified, this ExplainRequest will be served by the
      chosen DeployedModel, overriding Endpoint.traffic_split.
    explanationSpecOverride: If specified, overrides the explanation_spec of
      the DeployedModel. Can be used for explaining prediction results with
      different configurations, such as: - Explaining top-5 predictions
      results as opposed to top-1; - Increasing path count or step count of
      the attribution methods to reduce approximate errors; - Using different
      baselines for explaining the prediction results.
    instances: Required. The instances that are the input to the explanation
      call. A DeployedModel may have an upper limit on the number of instances
      it supports per request, and when it is exceeded the explanation call
      errors in case of AutoML Models, or, in case of customer created Models,
      the behaviour is as documented by that Model. The schema of any single
      instance may be specified via Endpoint's DeployedModels' Model's
      PredictSchemata's instance_schema_uri.
    parameters: The parameters that govern the prediction. The schema of the
      parameters may be specified via Endpoint's DeployedModels' Model's
      PredictSchemata's parameters_schema_uri.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ConcurrentExplanationSpecOverrideValue(_messages.Message):
        """Optional. This field is the same as the one above, but supports
    multiple explanations to occur in parallel. The key can be any string.
    Each override will be run against the model, then its explanations will be
    grouped together. Note - these explanations are run **In Addition** to the
    default Explanation in the deployed model.

    Messages:
      AdditionalProperty: An additional property for a
        ConcurrentExplanationSpecOverrideValue object.

    Fields:
      additionalProperties: Additional properties of type
        ConcurrentExplanationSpecOverrideValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ConcurrentExplanationSpecOverrideValue
      object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudAiplatformV1beta1ExplanationSpecOverride
          attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudAiplatformV1beta1ExplanationSpecOverride', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    concurrentExplanationSpecOverride = _messages.MessageField('ConcurrentExplanationSpecOverrideValue', 1)
    deployedModelId = _messages.StringField(2)
    explanationSpecOverride = _messages.MessageField('GoogleCloudAiplatformV1beta1ExplanationSpecOverride', 3)
    instances = _messages.MessageField('extra_types.JsonValue', 4, repeated=True)
    parameters = _messages.MessageField('extra_types.JsonValue', 5)