from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDatastoreAdminV1ImportEntitiesRequest(_messages.Message):
    """The request for google.datastore.admin.v1.DatastoreAdmin.ImportEntities.

  Messages:
    LabelsValue: Client-assigned labels.

  Fields:
    entityFilter: Optionally specify which kinds/namespaces are to be
      imported. If provided, the list must be a subset of the EntityFilter
      used in creating the export, otherwise a FAILED_PRECONDITION error will
      be returned. If no filter is specified then all entities from the export
      are imported.
    inputUrl: Required. The full resource URL of the external storage
      location. Currently, only Google Cloud Storage is supported. So
      input_url should be of the form:
      `gs://BUCKET_NAME[/NAMESPACE_PATH]/OVERALL_EXPORT_METADATA_FILE`, where
      `BUCKET_NAME` is the name of the Cloud Storage bucket, `NAMESPACE_PATH`
      is an optional Cloud Storage namespace path (this is not a Cloud
      Datastore namespace), and `OVERALL_EXPORT_METADATA_FILE` is the metadata
      file written by the ExportEntities operation. For more information about
      Cloud Storage namespace paths, see [Object name
      considerations](https://cloud.google.com/storage/docs/naming#object-
      considerations). For more information, see
      google.datastore.admin.v1.ExportEntitiesResponse.output_url.
    labels: Client-assigned labels.
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
    entityFilter = _messages.MessageField('GoogleDatastoreAdminV1EntityFilter', 1)
    inputUrl = _messages.StringField(2)
    labels = _messages.MessageField('LabelsValue', 3)