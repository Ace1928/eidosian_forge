from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Consent(_messages.Message):
    """Represents a user's consent.

  Enums:
    StateValueValuesEnum: Required. Indicates the current state of this
      Consent.

  Fields:
    consentArtifact: Required. The resource name of the Consent artifact that
      contains proof of the end user's consent, of the form `projects/{project
      _id}/locations/{location_id}/datasets/{dataset_id}/consentStores/{consen
      t_store_id}/consentArtifacts/{consent_artifact_id}`.
    name: Resource name of the Consent, of the form `projects/{project_id}/loc
      ations/{location_id}/datasets/{dataset_id}/consentStores/{consent_store_
      id}/consents/{consent_id}`. Cannot be changed after creation.
    policies: Optional. Represents a user's consent in terms of the resources
      that can be accessed and under what conditions.
    state: Required. Indicates the current state of this Consent.
    userId: Required. User's UUID provided by the client.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Required. Indicates the current state of this Consent.

    Values:
      STATE_UNSPECIFIED: No state specified. Treated as ACTIVE only at the
        time of resource creation.
      ACTIVE: The Consent is active and is considered when evaluating a user's
        consent on resources.
      ARCHIVED: The archived state is currently not being used.
      REVOKED: A revoked Consent is not considered when evaluating a user's
        consent on resources.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        ARCHIVED = 2
        REVOKED = 3
    consentArtifact = _messages.StringField(1)
    name = _messages.StringField(2)
    policies = _messages.MessageField('GoogleCloudHealthcareV1alpha2ConsentPolicy', 3, repeated=True)
    state = _messages.EnumField('StateValueValuesEnum', 4)
    userId = _messages.StringField(5)