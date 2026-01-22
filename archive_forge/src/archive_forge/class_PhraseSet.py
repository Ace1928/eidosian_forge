from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PhraseSet(_messages.Message):
    """PhraseSet for biasing in speech recognition. A PhraseSet is used to
  provide "hints" to the speech recognizer to favor specific words and phrases
  in the results.

  Enums:
    StateValueValuesEnum: Output only. The PhraseSet lifecycle state.

  Messages:
    AnnotationsValue: Allows users to store small amounts of arbitrary data.
      Both the key and the value must be 63 characters or less each. At most
      100 annotations.

  Fields:
    annotations: Allows users to store small amounts of arbitrary data. Both
      the key and the value must be 63 characters or less each. At most 100
      annotations.
    boost: Hint Boost. Positive value will increase the probability that a
      specific phrase will be recognized over other similar sounding phrases.
      The higher the boost, the higher the chance of false positive
      recognition as well. Valid `boost` values are between 0 (exclusive) and
      20. We recommend using a binary search approach to finding the optimal
      value for your use case as well as adding phrases both with and without
      boost to your requests.
    createTime: Output only. Creation time.
    deleteTime: Output only. The time at which this resource was requested for
      deletion.
    displayName: User-settable, human-readable name for the PhraseSet. Must be
      63 characters or less.
    etag: Output only. This checksum is computed by the server based on the
      value of other fields. This may be sent on update, undelete, and delete
      requests to ensure the client has an up-to-date value before proceeding.
    expireTime: Output only. The time at which this resource will be purged.
    kmsKeyName: Output only. The [KMS key
      name](https://cloud.google.com/kms/docs/resource-hierarchy#keys) with
      which the PhraseSet is encrypted. The expected format is `projects/{proj
      ect}/locations/{location}/keyRings/{key_ring}/cryptoKeys/{crypto_key}`.
    kmsKeyVersionName: Output only. The [KMS key version
      name](https://cloud.google.com/kms/docs/resource-hierarchy#key_versions)
      with which the PhraseSet is encrypted. The expected format is `projects/
      {project}/locations/{location}/keyRings/{key_ring}/cryptoKeys/{crypto_ke
      y}/cryptoKeyVersions/{crypto_key_version}`.
    name: Output only. Identifier. The resource name of the PhraseSet. Format:
      `projects/{project}/locations/{location}/phraseSets/{phrase_set}`.
    phrases: A list of word and phrases.
    reconciling: Output only. Whether or not this PhraseSet is in the process
      of being updated.
    state: Output only. The PhraseSet lifecycle state.
    uid: Output only. System-assigned unique identifier for the PhraseSet.
    updateTime: Output only. The most recent time this resource was modified.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The PhraseSet lifecycle state.

    Values:
      STATE_UNSPECIFIED: Unspecified state. This is only used/useful for
        distinguishing unset values.
      ACTIVE: The normal and active state.
      DELETED: This PhraseSet has been deleted.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        DELETED = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Allows users to store small amounts of arbitrary data. Both the key
    and the value must be 63 characters or less each. At most 100 annotations.

    Messages:
      AdditionalProperty: An additional property for a AnnotationsValue
        object.

    Fields:
      additionalProperties: Additional properties of type AnnotationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AnnotationsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    annotations = _messages.MessageField('AnnotationsValue', 1)
    boost = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    createTime = _messages.StringField(3)
    deleteTime = _messages.StringField(4)
    displayName = _messages.StringField(5)
    etag = _messages.StringField(6)
    expireTime = _messages.StringField(7)
    kmsKeyName = _messages.StringField(8)
    kmsKeyVersionName = _messages.StringField(9)
    name = _messages.StringField(10)
    phrases = _messages.MessageField('Phrase', 11, repeated=True)
    reconciling = _messages.BooleanField(12)
    state = _messages.EnumField('StateValueValuesEnum', 13)
    uid = _messages.StringField(14)
    updateTime = _messages.StringField(15)