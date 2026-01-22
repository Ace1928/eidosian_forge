from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TrustDirectionValueValuesEnum(_messages.Enum):
    """Required. The trust direction, which decides if the current domain is
    trusted, trusting, or both.

    Values:
      TRUST_DIRECTION_UNSPECIFIED: Not set.
      INBOUND: The inbound direction represents the trusting side.
      OUTBOUND: The outboud direction represents the trusted side.
      BIDIRECTIONAL: The bidirectional direction represents the trusted /
        trusting side.
    """
    TRUST_DIRECTION_UNSPECIFIED = 0
    INBOUND = 1
    OUTBOUND = 2
    BIDIRECTIONAL = 3