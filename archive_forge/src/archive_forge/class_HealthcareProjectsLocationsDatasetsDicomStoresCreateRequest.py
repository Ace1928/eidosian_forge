from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsDicomStoresCreateRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsDicomStoresCreateRequest object.

  Fields:
    dicomStore: A DicomStore resource to be passed as the request body.
    dicomStoreId: Required. The ID of the DICOM store that is being created.
      Any string value up to 256 characters in length.
    parent: Required. The name of the dataset this DICOM store belongs to.
  """
    dicomStore = _messages.MessageField('DicomStore', 1)
    dicomStoreId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)