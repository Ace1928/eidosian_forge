from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EndpointAttachment(_messages.Message):
    """represents the Connector's Endpoint Attachment resource

  Messages:
    LabelsValue: Optional. Resource labels to represent user-provided
      metadata. Refer to cloud documentation on labels for more details.
      https://cloud.google.com/compute/docs/labeling-resources

  Fields:
    createTime: Output only. Created time.
    description: Optional. Description of the resource.
    endpointIp: Output only. The Private Service Connect connection endpoint
      ip
    labels: Optional. Resource labels to represent user-provided metadata.
      Refer to cloud documentation on labels for more details.
      https://cloud.google.com/compute/docs/labeling-resources
    name: Output only. Resource name of the Endpoint Attachment. Format: proje
      cts/{project}/locations/{location}/endpointAttachments/{endpoint_attachm
      ent}
    serviceAttachment: Required. The path of the service attachment
    updateTime: Output only. Updated time.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Resource labels to represent user-provided metadata. Refer
    to cloud documentation on labels for more details.
    https://cloud.google.com/compute/docs/labeling-resources

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
    description = _messages.StringField(2)
    endpointIp = _messages.StringField(3)
    labels = _messages.MessageField('LabelsValue', 4)
    name = _messages.StringField(5)
    serviceAttachment = _messages.StringField(6)
    updateTime = _messages.StringField(7)