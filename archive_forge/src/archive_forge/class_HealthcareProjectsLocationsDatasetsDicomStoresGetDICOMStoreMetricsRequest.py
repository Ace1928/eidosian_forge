from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsDicomStoresGetDICOMStoreMetricsRequest(_messages.Message):
    """A
  HealthcareProjectsLocationsDatasetsDicomStoresGetDICOMStoreMetricsRequest
  object.

  Fields:
    name: Required. The resource name of the DICOM store to get metrics for.
  """
    name = _messages.StringField(1, required=True)