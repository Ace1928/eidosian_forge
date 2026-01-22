from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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