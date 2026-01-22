from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PlatformEditionValueValuesEnum(_messages.Enum):
    """Platform edition.

    Values:
      PLATFORM_EDITION_UNSPECIFIED: Platform edition is unspecified.
      LOOKER_CORE_TRIAL: Trial.
      LOOKER_CORE_STANDARD: Standard.
      LOOKER_CORE_STANDARD_ANNUAL: Subscription Standard.
      LOOKER_CORE_ENTERPRISE_ANNUAL: Subscription Enterprise.
      LOOKER_CORE_EMBED_ANNUAL: Subscription Embed.
    """
    PLATFORM_EDITION_UNSPECIFIED = 0
    LOOKER_CORE_TRIAL = 1
    LOOKER_CORE_STANDARD = 2
    LOOKER_CORE_STANDARD_ANNUAL = 3
    LOOKER_CORE_ENTERPRISE_ANNUAL = 4
    LOOKER_CORE_EMBED_ANNUAL = 5