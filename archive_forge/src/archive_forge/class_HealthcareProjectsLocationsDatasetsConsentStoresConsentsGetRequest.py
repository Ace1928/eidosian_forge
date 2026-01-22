from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsConsentStoresConsentsGetRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsConsentStoresConsentsGetRequest
  object.

  Fields:
    name: Required. The resource name of the Consent to retrieve, of the form
      `projects/{project_id}/locations/{location_id}/datasets/{dataset_id}/con
      sentStores/{consent_store_id}/consents/{consent_id}`. In order to
      retrieve a previous revision of the Consent, also provide the revision
      ID: `projects/{project_id}/locations/{location_id}/datasets/{dataset_id}
      /consentStores/{consent_store_id}/consents/{consent_id}@{revision_id}`
  """
    name = _messages.StringField(1, required=True)