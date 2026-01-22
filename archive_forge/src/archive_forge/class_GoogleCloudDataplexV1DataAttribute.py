from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataAttribute(_messages.Message):
    """Denotes one dataAttribute in a dataTaxonomy, for example, PII.
  DataAttribute resources can be defined in a hierarchy. A single
  dataAttribute resource can contain specs of multiple types PII -
  ResourceAccessSpec : - readers :foo@bar.com - DataAccessSpec : - readers
  :bar@foo.com

  Messages:
    LabelsValue: Optional. User-defined labels for the DataAttribute.

  Fields:
    attributeCount: Output only. The number of child attributes present for
      this attribute.
    createTime: Output only. The time when the DataAttribute was created.
    dataAccessSpec: Optional. Specified when applied to data stored on the
      resource (eg: rows, columns in BigQuery Tables).
    description: Optional. Description of the DataAttribute.
    displayName: Optional. User friendly display name.
    etag: This checksum is computed by the server based on the value of other
      fields, and may be sent on update and delete requests to ensure the
      client has an up-to-date value before proceeding.
    labels: Optional. User-defined labels for the DataAttribute.
    name: Output only. The relative resource name of the dataAttribute, of the
      form: projects/{project_number}/locations/{location_id}/dataTaxonomies/{
      dataTaxonomy}/attributes/{data_attribute_id}.
    parentId: Optional. The ID of the parent DataAttribute resource, should
      belong to the same data taxonomy. Circular dependency in parent chain is
      not valid. Maximum depth of the hierarchy allowed is 4. a -> b -> c -> d
      -> e, depth = 4
    resourceAccessSpec: Optional. Specified when applied to a resource (eg:
      Cloud Storage bucket, BigQuery dataset, BigQuery table).
    uid: Output only. System generated globally unique ID for the
      DataAttribute. This ID will be different if the DataAttribute is deleted
      and re-created with the same name.
    updateTime: Output only. The time when the DataAttribute was last updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. User-defined labels for the DataAttribute.

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
    attributeCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    createTime = _messages.StringField(2)
    dataAccessSpec = _messages.MessageField('GoogleCloudDataplexV1DataAccessSpec', 3)
    description = _messages.StringField(4)
    displayName = _messages.StringField(5)
    etag = _messages.StringField(6)
    labels = _messages.MessageField('LabelsValue', 7)
    name = _messages.StringField(8)
    parentId = _messages.StringField(9)
    resourceAccessSpec = _messages.MessageField('GoogleCloudDataplexV1ResourceAccessSpec', 10)
    uid = _messages.StringField(11)
    updateTime = _messages.StringField(12)