from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsDicomStoresDicomWebStudiesSeriesGetSeriesMetricsRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsDicomStoresDicomWebStudiesSeriesGet
  SeriesMetricsRequest object.

  Fields:
    series: The series resource path. For example, `projects/{project_id}/loca
      tions/{location_id}/datasets/{dataset_id}/dicomStores/{dicom_store_id}/d
      icomWeb/studies/{study_uid}/series/{series_uid}`.
  """
    series = _messages.StringField(1, required=True)