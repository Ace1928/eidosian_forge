from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsDicomStoresDicomWebStudiesSeriesInstancesGetStorageInfoRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsDicomStoresDicomWebStudiesSeriesIns
  tancesGetStorageInfoRequest object.

  Fields:
    resource: Required. The path of the resource for which the storage info is
      requested (for exaxmple for a DICOM Instance: `projects/{projectID}/loca
      tions/{locationID}/datasets/{datasetID}/dicomStores/{dicomStoreId}/dicom
      Web/studies/{study_uid}/series/{series_uid}/instances/{instance_uid}`)
  """
    resource = _messages.StringField(1, required=True)