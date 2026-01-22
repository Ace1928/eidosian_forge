from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsConsentStoresConsentsPatchRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsConsentStoresConsentsPatchRequest
  object.

  Fields:
    consent: A Consent resource to be passed as the request body.
    name: Resource name of the Consent, of the form `projects/{project_id}/loc
      ations/{location_id}/datasets/{dataset_id}/consentStores/{consent_store_
      id}/consents/{consent_id}`. Cannot be changed after creation.
    updateMask: Required. The update mask to apply to the resource. For the
      `FieldMask` definition, see https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask. Only the `user_id`,
      `policies`, `consent_artifact`, and `metadata` fields can be updated.
  """
    consent = _messages.MessageField('Consent', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)