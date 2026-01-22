from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataAttributeBinding(_messages.Message):
    """DataAttributeBinding represents binding of attributes to resources. Eg:
  Bind 'CustomerInfo' entity with 'PII' attribute.

  Messages:
    LabelsValue: Optional. User-defined labels for the DataAttributeBinding.

  Fields:
    attributes: Optional. List of attributes to be associated with the
      resource, provided in the form: projects/{project}/locations/{location}/
      dataTaxonomies/{dataTaxonomy}/attributes/{data_attribute_id}
    createTime: Output only. The time when the DataAttributeBinding was
      created.
    description: Optional. Description of the DataAttributeBinding.
    displayName: Optional. User friendly display name.
    etag: This checksum is computed by the server based on the value of other
      fields, and may be sent on update and delete requests to ensure the
      client has an up-to-date value before proceeding. Etags must be used
      when calling the DeleteDataAttributeBinding and the
      UpdateDataAttributeBinding method.
    labels: Optional. User-defined labels for the DataAttributeBinding.
    name: Output only. The relative resource name of the Data Attribute
      Binding, of the form: projects/{project_number}/locations/{location}/dat
      aAttributeBindings/{data_attribute_binding_id}
    paths: Optional. The list of paths for items within the associated
      resource (eg. columns and partitions within a table) along with
      attribute bindings.
    resource: Optional. Immutable. The resource name of the resource that is
      associated to attributes. Presently, only entity resource is supported
      in the form: projects/{project}/locations/{location}/lakes/{lake}/zones/
      {zone}/entities/{entity_id} Must belong in the same project and region
      as the attribute binding, and there can only exist one active binding
      for a resource.
    uid: Output only. System generated globally unique ID for the
      DataAttributeBinding. This ID will be different if the
      DataAttributeBinding is deleted and re-created with the same name.
    updateTime: Output only. The time when the DataAttributeBinding was last
      updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. User-defined labels for the DataAttributeBinding.

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
    attributes = _messages.StringField(1, repeated=True)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    displayName = _messages.StringField(4)
    etag = _messages.StringField(5)
    labels = _messages.MessageField('LabelsValue', 6)
    name = _messages.StringField(7)
    paths = _messages.MessageField('GoogleCloudDataplexV1DataAttributeBindingPath', 8, repeated=True)
    resource = _messages.StringField(9)
    uid = _messages.StringField(10)
    updateTime = _messages.StringField(11)