from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SSEGatewayReference(_messages.Message):
    """Message describing SSEGatewayReference object

  Messages:
    LabelsValue: Optional. Labels as key value pairs

  Fields:
    createTime: Output only. [Output only] Create time stamp
    labels: Optional. Labels as key value pairs
    name: Immutable. name of resource
    partnerSseRealm: Output only. PartnerSSERealm owning the PartnerSSEGateway
      that this SSEGateway intends to connect with
    proberSubnetRanges: Output only. Subnet ranges for Google probe packets.
    updateTime: Output only. [Output only] Update time stamp
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels as key value pairs

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
    createTime = _messages.StringField(1)
    labels = _messages.MessageField('LabelsValue', 2)
    name = _messages.StringField(3)
    partnerSseRealm = _messages.StringField(4)
    proberSubnetRanges = _messages.StringField(5, repeated=True)
    updateTime = _messages.StringField(6)