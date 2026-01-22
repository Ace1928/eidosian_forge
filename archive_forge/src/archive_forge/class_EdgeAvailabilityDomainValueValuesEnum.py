from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EdgeAvailabilityDomainValueValuesEnum(_messages.Enum):
    """Desired availability domain for the attachment. Only available for
    type PARTNER, at creation time, and can take one of the following values:
    - AVAILABILITY_DOMAIN_ANY - AVAILABILITY_DOMAIN_1 - AVAILABILITY_DOMAIN_2
    For improved reliability, customers should configure a pair of
    attachments, one per availability domain. The selected availability domain
    will be provided to the Partner via the pairing key, so that the
    provisioned circuit will lie in the specified domain. If not specified,
    the value will default to AVAILABILITY_DOMAIN_ANY.

    Values:
      AVAILABILITY_DOMAIN_1: <no description>
      AVAILABILITY_DOMAIN_2: <no description>
      AVAILABILITY_DOMAIN_ANY: <no description>
    """
    AVAILABILITY_DOMAIN_1 = 0
    AVAILABILITY_DOMAIN_2 = 1
    AVAILABILITY_DOMAIN_ANY = 2