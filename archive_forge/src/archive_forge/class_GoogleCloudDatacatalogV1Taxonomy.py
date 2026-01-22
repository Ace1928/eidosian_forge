from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1Taxonomy(_messages.Message):
    """A taxonomy is a collection of hierarchical policy tags that classify
  data along a common axis. For example, a "data sensitivity" taxonomy might
  contain the following policy tags: ``` + PII + Account number + Age + SSN +
  Zipcode + Financials + Revenue ``` A "data origin" taxonomy might contain
  the following policy tags: ``` + User data + Employee data + Partner data +
  Public data ```

  Enums:
    ActivatedPolicyTypesValueListEntryValuesEnum:

  Fields:
    activatedPolicyTypes: Optional. A list of policy types that are activated
      for this taxonomy. If not set, defaults to an empty list.
    description: Optional. Description of this taxonomy. If not set, defaults
      to empty. The description must contain only Unicode characters, tabs,
      newlines, carriage returns, and page breaks, and be at most 2000 bytes
      long when encoded in UTF-8.
    displayName: Required. User-defined name of this taxonomy. The name can't
      start or end with spaces, must contain only Unicode letters, numbers,
      underscores, dashes, and spaces, and be at most 200 bytes long when
      encoded in UTF-8. The taxonomy display name must be unique within an
      organization.
    name: Identifier. Resource name of this taxonomy in URL format. Note:
      Policy tag manager generates unique taxonomy IDs.
    policyTagCount: Output only. Number of policy tags in this taxonomy.
    service: Output only. Identity of the service which owns the Taxonomy.
      This field is only populated when the taxonomy is created by a Google
      Cloud service. Currently only 'DATAPLEX' is supported.
    taxonomyTimestamps: Output only. Creation and modification timestamps of
      this taxonomy.
  """

    class ActivatedPolicyTypesValueListEntryValuesEnum(_messages.Enum):
        """ActivatedPolicyTypesValueListEntryValuesEnum enum type.

    Values:
      POLICY_TYPE_UNSPECIFIED: Unspecified policy type.
      FINE_GRAINED_ACCESS_CONTROL: Fine-grained access control policy that
        enables access control on tagged sub-resources.
    """
        POLICY_TYPE_UNSPECIFIED = 0
        FINE_GRAINED_ACCESS_CONTROL = 1
    activatedPolicyTypes = _messages.EnumField('ActivatedPolicyTypesValueListEntryValuesEnum', 1, repeated=True)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    name = _messages.StringField(4)
    policyTagCount = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    service = _messages.MessageField('GoogleCloudDatacatalogV1TaxonomyService', 6)
    taxonomyTimestamps = _messages.MessageField('GoogleCloudDatacatalogV1SystemTimestamps', 7)