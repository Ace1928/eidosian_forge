from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsDicomStoresImportRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsDicomStoresImportRequest object.

  Fields:
    importDicomDataRequest: A ImportDicomDataRequest resource to be passed as
      the request body.
    name: Required. The name of the DICOM store resource into which the data
      is imported. For example, `projects/{project_id}/locations/{location_id}
      /datasets/{dataset_id}/dicomStores/{dicom_store_id}`.
  """
    importDicomDataRequest = _messages.MessageField('ImportDicomDataRequest', 1)
    name = _messages.StringField(2, required=True)