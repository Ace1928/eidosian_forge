from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SeriesMetrics(_messages.Message):
    """SeriesMetrics contains metrics describing a DICOM series.

  Fields:
    blobStorageSizeBytes: Total blob storage bytes for all instances in the
      series.
    instanceCount: Number of instances in the series.
    series: The series resource path. For example, `projects/{project_id}/loca
      tions/{location_id}/datasets/{dataset_id}/dicomStores/{dicom_store_id}/d
      icomWeb/studies/{study_uid}/series/{series_uid}`.
    structuredStorageSizeBytes: Total structured storage bytes for all
      instances in the series.
  """
    blobStorageSizeBytes = _messages.IntegerField(1)
    instanceCount = _messages.IntegerField(2)
    series = _messages.StringField(3)
    structuredStorageSizeBytes = _messages.IntegerField(4)