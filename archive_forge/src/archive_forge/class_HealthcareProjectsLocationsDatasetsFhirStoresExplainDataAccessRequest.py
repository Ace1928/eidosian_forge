from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresExplainDataAccessRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsFhirStoresExplainDataAccessRequest
  object.

  Fields:
    name: Required. The name of the FHIR store to enforce, in the format `proj
      ects/{project_id}/locations/{location_id}/datasets/{dataset_id}/fhirStor
      es/{fhir_store_id}`.
    resourceId: Required. The ID (`{resourceType}/{id}`) of the resource to
      explain data access on.
  """
    name = _messages.StringField(1, required=True)
    resourceId = _messages.StringField(2)