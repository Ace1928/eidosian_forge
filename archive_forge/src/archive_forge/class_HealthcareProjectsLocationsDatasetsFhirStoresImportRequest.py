from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresImportRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsFhirStoresImportRequest object.

  Fields:
    importResourcesRequest: A ImportResourcesRequest resource to be passed as
      the request body.
    name: Required. The name of the FHIR store to which the server imports
      FHIR resources, in the format `projects/{project_id}/locations/{location
      _id}/datasets/{dataset_id}/fhirStores/{fhir_store_id}`.
  """
    importResourcesRequest = _messages.MessageField('ImportResourcesRequest', 1)
    name = _messages.StringField(2, required=True)