from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetastoreProjectsLocationsServicesExportMetadataRequest(_messages.Message):
    """A MetastoreProjectsLocationsServicesExportMetadataRequest object.

  Fields:
    exportMetadataRequest: A ExportMetadataRequest resource to be passed as
      the request body.
    service: Required. The relative resource name of the metastore service to
      run export, in the following form:projects/{project_id}/locations/{locat
      ion_id}/services/{service_id}.
  """
    exportMetadataRequest = _messages.MessageField('ExportMetadataRequest', 1)
    service = _messages.StringField(2, required=True)