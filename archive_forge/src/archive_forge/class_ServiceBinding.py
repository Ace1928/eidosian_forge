from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceBinding(_messages.Message):
    """ServiceBinding is the resource that defines a Service Directory Service
  to be used in a BackendService resource.

  Messages:
    LabelsValue: Optional. Set of label tags associated with the
      ServiceBinding resource.

  Fields:
    createTime: Output only. The timestamp when the resource was created.
    description: Optional. A free-text description of the resource. Max length
      1024 characters.
    labels: Optional. Set of label tags associated with the ServiceBinding
      resource.
    name: Required. Name of the ServiceBinding resource. It matches pattern
      `projects/*/locations/global/serviceBindings/service_binding_name`.
    service: Required. The full Service Directory Service name of the format
      projects/*/locations/*/namespaces/*/services/*
    serviceId: Output only. The unique identifier of the Service Directory
      Service against which the Service Binding resource is validated. This is
      populated when the Service Binding resource is used in another resource
      (like Backend Service). This is of the UUID4 format.
    updateTime: Output only. The timestamp when the resource was updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Set of label tags associated with the ServiceBinding
    resource.

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
    labels = _messages.MessageField('LabelsValue', 3)
    name = _messages.StringField(4)
    service = _messages.StringField(5)
    serviceId = _messages.StringField(6)
    updateTime = _messages.StringField(7)