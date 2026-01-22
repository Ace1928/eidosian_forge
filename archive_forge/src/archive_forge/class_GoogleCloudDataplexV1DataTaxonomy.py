from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataTaxonomy(_messages.Message):
    """DataTaxonomy represents a set of hierarchical DataAttributes resources,
  grouped with a common theme Eg: 'SensitiveDataTaxonomy' can have attributes
  to manage PII data. It is defined at project level.

  Messages:
    LabelsValue: Optional. User-defined labels for the DataTaxonomy.

  Fields:
    attributeCount: Output only. The number of attributes in the DataTaxonomy.
    classCount: Output only. The number of classes in the DataTaxonomy.
    createTime: Output only. The time when the DataTaxonomy was created.
    description: Optional. Description of the DataTaxonomy.
    displayName: Optional. User friendly display name.
    etag: This checksum is computed by the server based on the value of other
      fields, and may be sent on update and delete requests to ensure the
      client has an up-to-date value before proceeding.
    labels: Optional. User-defined labels for the DataTaxonomy.
    name: Output only. The relative resource name of the DataTaxonomy, of the
      form: projects/{project_number}/locations/{location_id}/dataTaxonomies/{
      data_taxonomy_id}.
    uid: Output only. System generated globally unique ID for the
      dataTaxonomy. This ID will be different if the DataTaxonomy is deleted
      and re-created with the same name.
    updateTime: Output only. The time when the DataTaxonomy was last updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. User-defined labels for the DataTaxonomy.

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
    classCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    createTime = _messages.StringField(3)
    description = _messages.StringField(4)
    displayName = _messages.StringField(5)
    etag = _messages.StringField(6)
    labels = _messages.MessageField('LabelsValue', 7)
    name = _messages.StringField(8)
    uid = _messages.StringField(9)
    updateTime = _messages.StringField(10)