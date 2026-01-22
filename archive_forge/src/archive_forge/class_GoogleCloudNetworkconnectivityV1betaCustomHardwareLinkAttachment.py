from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudNetworkconnectivityV1betaCustomHardwareLinkAttachment(_messages.Message):
    """Message describing CustomHardwareLinkAttachment object

  Enums:
    LinkTypeValueValuesEnum: Required. The type of custom hardware link
      attachment.

  Messages:
    LabelsValue: Optional. User-defined labels.

  Fields:
    createTime: Output only. Time when the CustomHardwareLinkAttachment was
      created.
    labels: Optional. User-defined labels.
    linkType: Required. The type of custom hardware link attachment.
    name: Identifier. The name of a CustomHardwareLinkAttachment. Format: `pro
      jects/{project}/locations/{location}/customHardwareLinkAttachments/{cust
      om_hardware_link_attachment}`.
    network: The name of the VPC network for this custom hardware link
      attachment. Format: `projects/{project}/global/networks/{network}`
    project: The consumer project where custom hardware instance are created.
      Format: `projects/{project}`
    updateTime: Output only. Time when the CustomHardwareLinkAttachment was
      updated.
  """

    class LinkTypeValueValuesEnum(_messages.Enum):
        """Required. The type of custom hardware link attachment.

    Values:
      LINK_TYPE_UNSPECIFIED: An invalid type as the default case.
      REGULAR: Regular traffic type.
      ULL: Ultra-low latency traffic type.
    """
        LINK_TYPE_UNSPECIFIED = 0
        REGULAR = 1
        ULL = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. User-defined labels.

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
    linkType = _messages.EnumField('LinkTypeValueValuesEnum', 3)
    name = _messages.StringField(4)
    network = _messages.StringField(5)
    project = _messages.StringField(6)
    updateTime = _messages.StringField(7)