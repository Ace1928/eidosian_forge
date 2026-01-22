from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDomainsDns(_messages.Message):
    """Deprecated: For more information, see [Cloud Domains feature
  deprecation](https://cloud.google.com/domains/docs/deprecations/feature-
  deprecations). Configuration for using the free DNS zone provided by Google
  Domains as a `Registration`'s `dns_provider`. You cannot configure the DNS
  zone itself using the API. To configure the DNS zone, go to [Google
  Domains](https://domains.google/).

  Enums:
    DsStateValueValuesEnum: Required. The state of DS records for this domain.
      Used to enable or disable automatic DNSSEC.

  Fields:
    dsRecords: Output only. The list of DS records published for this domain.
      The list is automatically populated when `ds_state` is
      `DS_RECORDS_PUBLISHED`, otherwise it remains empty.
    dsState: Required. The state of DS records for this domain. Used to enable
      or disable automatic DNSSEC.
    nameServers: Output only. A list of name servers that store the DNS zone
      for this domain. Each name server is a domain name, with Unicode domain
      names expressed in Punycode format. This field is automatically
      populated with the name servers assigned to the Google Domains DNS zone.
  """

    class DsStateValueValuesEnum(_messages.Enum):
        """Required. The state of DS records for this domain. Used to enable or
    disable automatic DNSSEC.

    Values:
      DS_STATE_UNSPECIFIED: DS state is unspecified.
      DS_RECORDS_UNPUBLISHED: DNSSEC is disabled for this domain. No DS
        records for this domain are published in the parent DNS zone.
      DS_RECORDS_PUBLISHED: DNSSEC is enabled for this domain. Appropriate DS
        records for this domain are published in the parent DNS zone. This
        option is valid only if the DNS zone referenced in the
        `Registration`'s `dns_provider` field is already DNSSEC-signed.
    """
        DS_STATE_UNSPECIFIED = 0
        DS_RECORDS_UNPUBLISHED = 1
        DS_RECORDS_PUBLISHED = 2
    dsRecords = _messages.MessageField('DsRecord', 1, repeated=True)
    dsState = _messages.EnumField('DsStateValueValuesEnum', 2)
    nameServers = _messages.StringField(3, repeated=True)