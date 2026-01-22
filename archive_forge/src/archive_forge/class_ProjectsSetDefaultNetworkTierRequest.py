from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProjectsSetDefaultNetworkTierRequest(_messages.Message):
    """A ProjectsSetDefaultNetworkTierRequest object.

  Enums:
    NetworkTierValueValuesEnum: Default network tier to be set.

  Fields:
    networkTier: Default network tier to be set.
  """

    class NetworkTierValueValuesEnum(_messages.Enum):
        """Default network tier to be set.

    Values:
      FIXED_STANDARD: Public internet quality with fixed bandwidth.
      PREMIUM: High quality, Google-grade network tier, support for all
        networking products.
      STANDARD: Public internet quality, only limited support for other
        networking products.
      STANDARD_OVERRIDES_FIXED_STANDARD: (Output only) Temporary tier for
        FIXED_STANDARD when fixed standard tier is expired or not configured.
    """
        FIXED_STANDARD = 0
        PREMIUM = 1
        STANDARD = 2
        STANDARD_OVERRIDES_FIXED_STANDARD = 3
    networkTier = _messages.EnumField('NetworkTierValueValuesEnum', 1)