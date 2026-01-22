from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkTierValueValuesEnum(_messages.Enum):
    """Optional. This signifies the networking tier used for configuring this
    access

    Values:
      NETWORK_TIER_UNSPECIFIED: Default value. This value is unused.
      PREMIUM: High quality, Google-grade network tier, support for all
        networking products.
      STANDARD: Public internet quality, only limited support for other
        networking products.
    """
    NETWORK_TIER_UNSPECIFIED = 0
    PREMIUM = 1
    STANDARD = 2