from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConsentArtifact(_messages.Message):
    """Documentation of a user's consent.

  Messages:
    MetadataValue: Optional. Metadata associated with the Consent artifact.
      For example, the consent locale or user agent version.

  Fields:
    consentContentScreenshots: Optional. Screenshots, PDFs, or other binary
      information documenting the user's consent.
    consentContentVersion: Optional. An string indicating the version of the
      consent information shown to the user.
    guardianSignature: Optional. A signature from a guardian.
    metadata: Optional. Metadata associated with the Consent artifact. For
      example, the consent locale or user agent version.
    name: Identifier. Resource name of the Consent artifact, of the form `proj
      ects/{project_id}/locations/{location_id}/datasets/{dataset_id}/consentS
      tores/{consent_store_id}/consentArtifacts/{consent_artifact_id}`. Cannot
      be changed after creation.
    userId: Required. User's UUID provided by the client.
    userSignature: Optional. User's signature.
    witnessSignature: Optional. A signature from a witness.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetadataValue(_messages.Message):
        """Optional. Metadata associated with the Consent artifact. For example,
    the consent locale or user agent version.

    Messages:
      AdditionalProperty: An additional property for a MetadataValue object.

    Fields:
      additionalProperties: Additional properties of type MetadataValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    consentContentScreenshots = _messages.MessageField('Image', 1, repeated=True)
    consentContentVersion = _messages.StringField(2)
    guardianSignature = _messages.MessageField('Signature', 3)
    metadata = _messages.MessageField('MetadataValue', 4)
    name = _messages.StringField(5)
    userId = _messages.StringField(6)
    userSignature = _messages.MessageField('Signature', 7)
    witnessSignature = _messages.MessageField('Signature', 8)