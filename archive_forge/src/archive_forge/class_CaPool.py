from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CaPool(_messages.Message):
    """A CaPool represents a group of CertificateAuthorities that form a trust
  anchor. A CaPool can be used to manage issuance policies for one or more
  CertificateAuthority resources and to rotate CA certificates in and out of
  the trust anchor.

  Enums:
    TierValueValuesEnum: Required. Immutable. The Tier of this CaPool.

  Messages:
    LabelsValue: Optional. Labels with user-defined metadata.

  Fields:
    issuancePolicy: Optional. The IssuancePolicy to control how Certificates
      will be issued from this CaPool.
    labels: Optional. Labels with user-defined metadata.
    name: Output only. The resource name for this CaPool in the format
      `projects/*/locations/*/caPools/*`.
    publishingOptions: Optional. The PublishingOptions to follow when issuing
      Certificates from any CertificateAuthority in this CaPool.
    tier: Required. Immutable. The Tier of this CaPool.
  """

    class TierValueValuesEnum(_messages.Enum):
        """Required. Immutable. The Tier of this CaPool.

    Values:
      TIER_UNSPECIFIED: Not specified.
      ENTERPRISE: Enterprise tier.
      DEVOPS: DevOps tier.
    """
        TIER_UNSPECIFIED = 0
        ENTERPRISE = 1
        DEVOPS = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels with user-defined metadata.

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
    issuancePolicy = _messages.MessageField('IssuancePolicy', 1)
    labels = _messages.MessageField('LabelsValue', 2)
    name = _messages.StringField(3)
    publishingOptions = _messages.MessageField('PublishingOptions', 4)
    tier = _messages.EnumField('TierValueValuesEnum', 5)