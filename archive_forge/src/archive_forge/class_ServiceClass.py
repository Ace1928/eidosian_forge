from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceClass(_messages.Message):
    """The ServiceClass resource. Next id: 9

  Messages:
    LabelsValue: User-defined labels.

  Fields:
    createTime: Output only. Time when the ServiceClass was created.
    description: A description of this resource.
    etag: Optional. The etag is computed by the server, and may be sent on
      update and delete requests to ensure the client has an up-to-date value
      before proceeding.
    labels: User-defined labels.
    name: Immutable. The name of a ServiceClass resource. Format:
      projects/{project}/locations/{location}/serviceClasses/{service_class}
      See: https://google.aip.dev/122#fields-representing-resource-names
    serviceClass: Output only. The generated service class name. Use this name
      to refer to the Service class in Service Connection Maps and Service
      Connection Policies.
    updateTime: Output only. Time when the ServiceClass was updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """User-defined labels.

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
    etag = _messages.StringField(3)
    labels = _messages.MessageField('LabelsValue', 4)
    name = _messages.StringField(5)
    serviceClass = _messages.StringField(6)
    updateTime = _messages.StringField(7)