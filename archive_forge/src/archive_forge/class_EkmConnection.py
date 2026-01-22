from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EkmConnection(_messages.Message):
    """An EkmConnection represents an individual EKM connection. It can be used
  for creating CryptoKeys and CryptoKeyVersions with a ProtectionLevel of
  EXTERNAL_VPC, as well as performing cryptographic operations using keys
  created within the EkmConnection.

  Enums:
    KeyManagementModeValueValuesEnum: Optional. Describes who can perform
      control plane operations on the EKM. If unset, this defaults to MANUAL.

  Fields:
    createTime: Output only. The time at which the EkmConnection was created.
    cryptoSpacePath: Optional. Identifies the EKM Crypto Space that this
      EkmConnection maps to. Note: This field is required if KeyManagementMode
      is CLOUD_KMS.
    etag: Optional. Etag of the currently stored EkmConnection.
    keyManagementMode: Optional. Describes who can perform control plane
      operations on the EKM. If unset, this defaults to MANUAL.
    name: Output only. The resource name for the EkmConnection in the format
      `projects/*/locations/*/ekmConnections/*`.
    serviceResolvers: A list of ServiceResolvers where the EKM can be reached.
      There should be one ServiceResolver per EKM replica. Currently, only a
      single ServiceResolver is supported.
  """

    class KeyManagementModeValueValuesEnum(_messages.Enum):
        """Optional. Describes who can perform control plane operations on the
    EKM. If unset, this defaults to MANUAL.

    Values:
      KEY_MANAGEMENT_MODE_UNSPECIFIED: Not specified.
      MANUAL: EKM-side key management operations on CryptoKeys created with
        this EkmConnection must be initiated from the EKM directly and cannot
        be performed from Cloud KMS. This means that: * When creating a
        CryptoKeyVersion associated with this EkmConnection, the caller must
        supply the key path of pre-existing external key material that will be
        linked to the CryptoKeyVersion. * Destruction of external key material
        cannot be requested via the Cloud KMS API and must be performed
        directly in the EKM. * Automatic rotation of key material is not
        supported.
      CLOUD_KMS: All CryptoKeys created with this EkmConnection use EKM-side
        key management operations initiated from Cloud KMS. This means that: *
        When a CryptoKeyVersion associated with this EkmConnection is created,
        the EKM automatically generates new key material and a new key path.
        The caller cannot supply the key path of pre-existing external key
        material. * Destruction of external key material associated with this
        EkmConnection can be requested by calling DestroyCryptoKeyVersion. *
        Automatic rotation of key material is supported.
    """
        KEY_MANAGEMENT_MODE_UNSPECIFIED = 0
        MANUAL = 1
        CLOUD_KMS = 2
    createTime = _messages.StringField(1)
    cryptoSpacePath = _messages.StringField(2)
    etag = _messages.StringField(3)
    keyManagementMode = _messages.EnumField('KeyManagementModeValueValuesEnum', 4)
    name = _messages.StringField(5)
    serviceResolvers = _messages.MessageField('ServiceResolver', 6, repeated=True)