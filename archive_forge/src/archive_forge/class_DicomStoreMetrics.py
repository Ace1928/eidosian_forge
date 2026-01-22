from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DicomStoreMetrics(_messages.Message):
    """DicomStoreMetrics contains metrics describing a DICOM store.

  Fields:
    blobStorageSizeBytes: Total blob storage bytes for all instances in the
      store.
    instanceCount: Number of instances in the store.
    name: Resource name of the DICOM store, of the form `projects/{project_id}
      /locations/{location_id}/datasets/{dataset_id}/dicomStores/{dicom_store_
      id}`.
    seriesCount: Number of series in the store.
    structuredStorageSizeBytes: Total structured storage bytes for all
      instances in the store.
    studyCount: Number of studies in the store.
  """
    blobStorageSizeBytes = _messages.IntegerField(1)
    instanceCount = _messages.IntegerField(2)
    name = _messages.StringField(3)
    seriesCount = _messages.IntegerField(4)
    structuredStorageSizeBytes = _messages.IntegerField(5)
    studyCount = _messages.IntegerField(6)