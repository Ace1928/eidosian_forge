from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsDicomStoresStudiesStoreInstancesRequest(_messages.Message):
    """A
  HealthcareProjectsLocationsDatasetsDicomStoresStudiesStoreInstancesRequest
  object.

  Fields:
    dicomWebPath: The path of the StoreInstances DICOMweb request. For
      example, `studies/[{study_uid}]`. Note that the `study_uid` is optional.
    httpBody: A HttpBody resource to be passed as the request body.
    parent: The name of the DICOM store that is being accessed. For example, `
      projects/{project_id}/locations/{location_id}/datasets/{dataset_id}/dico
      mStores/{dicom_store_id}`.
  """
    dicomWebPath = _messages.StringField(1, required=True)
    httpBody = _messages.MessageField('HttpBody', 2)
    parent = _messages.StringField(3, required=True)