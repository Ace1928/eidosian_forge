from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificateIssuanceConfig(_messages.Message):
    """CertificateIssuanceConfig specifies how to issue and manage a
  certificate.

  Enums:
    KeyAlgorithmValueValuesEnum: Required. The key algorithm to use when
      generating the private key.

  Messages:
    LabelsValue: Set of labels associated with a CertificateIssuanceConfig.

  Fields:
    certificateAuthorityConfig: Required. The CA that issues the workload
      certificate. It includes the CA address, type, authentication to CA
      service, etc.
    createTime: Output only. The creation timestamp of a
      CertificateIssuanceConfig.
    description: One or more paragraphs of text description of a
      CertificateIssuanceConfig.
    keyAlgorithm: Required. The key algorithm to use when generating the
      private key.
    labels: Set of labels associated with a CertificateIssuanceConfig.
    lifetime: Required. Workload certificate lifetime requested.
    name: A user-defined name of the certificate issuance config.
      CertificateIssuanceConfig names must be unique globally and match
      pattern `projects/*/locations/*/certificateIssuanceConfigs/*`.
    rotationWindowPercentage: Required. Specifies the percentage of elapsed
      time of the certificate lifetime to wait before renewing the
      certificate. Must be a number between 1-99, inclusive.
    updateTime: Output only. The last update timestamp of a
      CertificateIssuanceConfig.
  """

    class KeyAlgorithmValueValuesEnum(_messages.Enum):
        """Required. The key algorithm to use when generating the private key.

    Values:
      KEY_ALGORITHM_UNSPECIFIED: Unspecified key algorithm.
      RSA_2048: Specifies RSA with a 2048-bit modulus.
      ECDSA_P256: Specifies ECDSA with curve P256.
    """
        KEY_ALGORITHM_UNSPECIFIED = 0
        RSA_2048 = 1
        ECDSA_P256 = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Set of labels associated with a CertificateIssuanceConfig.

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
    certificateAuthorityConfig = _messages.MessageField('CertificateAuthorityConfig', 1)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    keyAlgorithm = _messages.EnumField('KeyAlgorithmValueValuesEnum', 4)
    labels = _messages.MessageField('LabelsValue', 5)
    lifetime = _messages.StringField(6)
    name = _messages.StringField(7)
    rotationWindowPercentage = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    updateTime = _messages.StringField(9)