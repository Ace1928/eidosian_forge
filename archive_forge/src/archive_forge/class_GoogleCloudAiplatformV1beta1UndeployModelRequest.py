from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1UndeployModelRequest(_messages.Message):
    """Request message for EndpointService.UndeployModel.

  Messages:
    TrafficSplitValue: If this field is provided, then the Endpoint's
      traffic_split will be overwritten with it. If last DeployedModel is
      being undeployed from the Endpoint, the [Endpoint.traffic_split] will
      always end up empty when this call returns. A DeployedModel will be
      successfully undeployed only if it doesn't have any traffic assigned to
      it when this method executes, or if this field unassigns any traffic to
      it.

  Fields:
    deployedModelId: Required. The ID of the DeployedModel to be undeployed
      from the Endpoint.
    trafficSplit: If this field is provided, then the Endpoint's traffic_split
      will be overwritten with it. If last DeployedModel is being undeployed
      from the Endpoint, the [Endpoint.traffic_split] will always end up empty
      when this call returns. A DeployedModel will be successfully undeployed
      only if it doesn't have any traffic assigned to it when this method
      executes, or if this field unassigns any traffic to it.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class TrafficSplitValue(_messages.Message):
        """If this field is provided, then the Endpoint's traffic_split will be
    overwritten with it. If last DeployedModel is being undeployed from the
    Endpoint, the [Endpoint.traffic_split] will always end up empty when this
    call returns. A DeployedModel will be successfully undeployed only if it
    doesn't have any traffic assigned to it when this method executes, or if
    this field unassigns any traffic to it.

    Messages:
      AdditionalProperty: An additional property for a TrafficSplitValue
        object.

    Fields:
      additionalProperties: Additional properties of type TrafficSplitValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a TrafficSplitValue object.

      Fields:
        key: Name of the additional property.
        value: A integer attribute.
      """
            key = _messages.StringField(1)
            value = _messages.IntegerField(2, variant=_messages.Variant.INT32)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    deployedModelId = _messages.StringField(1)
    trafficSplit = _messages.MessageField('TrafficSplitValue', 2)