from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDatastoreAdminV1beta1ExportEntitiesRequest(_messages.Message):
    """The request for
  google.datastore.admin.v1beta1.DatastoreAdmin.ExportEntities.

  Messages:
    LabelsValue: Client-assigned labels.

  Fields:
    entityFilter: Description of what data from the project is included in the
      export.
    labels: Client-assigned labels.
    outputUrlPrefix: Location for the export metadata and data files. The full
      resource URL of the external storage location. Currently, only Google
      Cloud Storage is supported. So output_url_prefix should be of the form:
      `gs://BUCKET_NAME[/NAMESPACE_PATH]`, where `BUCKET_NAME` is the name of
      the Cloud Storage bucket and `NAMESPACE_PATH` is an optional Cloud
      Storage namespace path (this is not a Cloud Datastore namespace). For
      more information about Cloud Storage namespace paths, see [Object name
      considerations](https://cloud.google.com/storage/docs/naming#object-
      considerations). The resulting files will be nested deeper than the
      specified URL prefix. The final output URL will be provided in the
      google.datastore.admin.v1beta1.ExportEntitiesResponse.output_url field.
      That value should be used for subsequent ImportEntities operations. By
      nesting the data files deeper, the same Cloud Storage bucket can be used
      in multiple ExportEntities operations without conflict.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Client-assigned labels.

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
    entityFilter = _messages.MessageField('GoogleDatastoreAdminV1beta1EntityFilter', 1)
    labels = _messages.MessageField('LabelsValue', 2)
    outputUrlPrefix = _messages.StringField(3)