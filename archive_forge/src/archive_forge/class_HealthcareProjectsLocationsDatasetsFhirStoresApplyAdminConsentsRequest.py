from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresApplyAdminConsentsRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsFhirStoresApplyAdminConsentsRequest
  object.

  Fields:
    applyAdminConsentsRequest: A ApplyAdminConsentsRequest resource to be
      passed as the request body.
    name: Required. The name of the FHIR store to enforce, in the format `proj
      ects/{project_id}/locations/{location_id}/datasets/{dataset_id}/fhirStor
      es/{fhir_store_id}`.
  """
    applyAdminConsentsRequest = _messages.MessageField('ApplyAdminConsentsRequest', 1)
    name = _messages.StringField(2, required=True)