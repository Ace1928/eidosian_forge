from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeliveryPipelineAttribute(_messages.Message):
    """Contains criteria for selecting DeliveryPipelines.

  Messages:
    LabelsValue: DeliveryPipeline labels.

  Fields:
    id: ID of the `DeliveryPipeline`. The value of this field could be one of
      the following: * The last segment of a pipeline name. It only needs the
      ID to determine which pipeline is being referred to * "*", all delivery
      pipelines in a location.
    labels: DeliveryPipeline labels.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """DeliveryPipeline labels.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    id = _messages.StringField(1)
    labels = _messages.MessageField('LabelsValue', 2)