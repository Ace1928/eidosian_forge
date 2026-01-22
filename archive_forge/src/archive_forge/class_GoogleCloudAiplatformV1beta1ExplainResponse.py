from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ExplainResponse(_messages.Message):
    """Response message for PredictionService.Explain.

  Messages:
    ConcurrentExplanationsValue: This field stores the results of the
      explanations run in parallel with The default explanation
      strategy/method.

  Fields:
    concurrentExplanations: This field stores the results of the explanations
      run in parallel with The default explanation strategy/method.
    deployedModelId: ID of the Endpoint's DeployedModel that served this
      explanation.
    explanations: The explanations of the Model's PredictResponse.predictions.
      It has the same number of elements as instances to be explained.
    predictions: The predictions that are the output of the predictions call.
      Same as PredictResponse.predictions.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ConcurrentExplanationsValue(_messages.Message):
        """This field stores the results of the explanations run in parallel with
    The default explanation strategy/method.

    Messages:
      AdditionalProperty: An additional property for a
        ConcurrentExplanationsValue object.

    Fields:
      additionalProperties: Additional properties of type
        ConcurrentExplanationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ConcurrentExplanationsValue object.

      Fields:
        key: Name of the additional property.
        value: A
          GoogleCloudAiplatformV1beta1ExplainResponseConcurrentExplanation
          attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudAiplatformV1beta1ExplainResponseConcurrentExplanation', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    concurrentExplanations = _messages.MessageField('ConcurrentExplanationsValue', 1)
    deployedModelId = _messages.StringField(2)
    explanations = _messages.MessageField('GoogleCloudAiplatformV1beta1Explanation', 3, repeated=True)
    predictions = _messages.MessageField('extra_types.JsonValue', 4, repeated=True)