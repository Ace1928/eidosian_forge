from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KmsConfig(_messages.Message):
    """KmsConfig is the customer managed encryption key(CMEK) configuration.

  Enums:
    StateValueValuesEnum: Output only. State of the KmsConfig.

  Messages:
    LabelsValue: Labels as key value pairs

  Fields:
    createTime: Output only. Create time of the KmsConfig.
    cryptoKeyName: Required. Customer managed crypto key resource full name.
      Format: projects/{project}/locations/{location}/keyRings/{key_ring}/cryp
      toKeys/{key}.
    description: Description of the KmsConfig.
    instructions: Output only. Instructions to provide the access to the
      customer provided encryption key.
    labels: Labels as key value pairs
    name: Identifier. Name of the KmsConfig.
    serviceAccount: Output only. The Service account which will have access to
      the customer provided encryption key.
    state: Output only. State of the KmsConfig.
    stateDetails: Output only. State details of the KmsConfig.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the KmsConfig.

    Values:
      STATE_UNSPECIFIED: Unspecified KmsConfig State
      READY: KmsConfig State is Ready
      CREATING: KmsConfig State is Creating
      DELETING: KmsConfig State is Deleting
      UPDATING: KmsConfig State is Updating
      IN_USE: KmsConfig State is In Use.
      ERROR: KmsConfig State is Error
      KEY_CHECK_PENDING: KmsConfig State is Pending to verify crypto key
        access.
      KEY_NOT_REACHABLE: KmsConfig State is Not accessbile by the SDE service
        account to the crypto key.
      DISABLING: KmsConfig State is Disabling.
      DISABLED: KmsConfig State is Disabled.
      MIGRATING: KmsConfig State is Migrating. The existing volumes are
        migrating from SMEK to CMEK.
    """
        STATE_UNSPECIFIED = 0
        READY = 1
        CREATING = 2
        DELETING = 3
        UPDATING = 4
        IN_USE = 5
        ERROR = 6
        KEY_CHECK_PENDING = 7
        KEY_NOT_REACHABLE = 8
        DISABLING = 9
        DISABLED = 10
        MIGRATING = 11

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels as key value pairs

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    cryptoKeyName = _messages.StringField(2)
    description = _messages.StringField(3)
    instructions = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    name = _messages.StringField(6)
    serviceAccount = _messages.StringField(7)
    state = _messages.EnumField('StateValueValuesEnum', 8)
    stateDetails = _messages.StringField(9)