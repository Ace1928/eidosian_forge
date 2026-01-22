from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsConsentStoresConsentsDeleteRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsConsentStoresConsentsDeleteRequest
  object.

  Fields:
    name: Required. The resource name of the Consent to delete, of the form `p
      rojects/{project_id}/locations/{location_id}/datasets/{dataset_id}/conse
      ntStores/{consent_store_id}/consents/{consent_id}`. An INVALID_ARGUMENT
      error occurs if `revision_id` is specified in the name.
  """
    name = _messages.StringField(1, required=True)