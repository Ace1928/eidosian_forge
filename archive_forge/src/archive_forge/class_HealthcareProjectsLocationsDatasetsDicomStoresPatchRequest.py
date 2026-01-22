from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsDicomStoresPatchRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsDicomStoresPatchRequest object.

  Fields:
    dicomStore: A DicomStore resource to be passed as the request body.
    name: Identifier. Resource name of the DICOM store, of the form `projects/
      {project_id}/locations/{location_id}/datasets/{dataset_id}/dicomStores/{
      dicom_store_id}`.
    updateMask: Required. The update mask applies to the resource. For the
      `FieldMask` definition, see https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask
  """
    dicomStore = _messages.MessageField('DicomStore', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)