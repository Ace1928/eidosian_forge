from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SafeBrowsingProtectionLevelValueValuesEnum(_messages.Enum):
    """Current state of [Safe Browsing protection level](https://chromeenterp
    rise.google/policies/#SafeBrowsingProtectionLevel).

    Values:
      SAFE_BROWSING_LEVEL_UNSPECIFIED: Browser protection level is not
        specified.
      DISABLED: No protection against dangerous websites, downloads, and
        extensions.
      STANDARD: Standard protection against websites, downloads, and
        extensions that are known to be dangerous.
      ENHANCED: Faster, proactive protection against dangerous websites,
        downloads, and extensions.
    """
    SAFE_BROWSING_LEVEL_UNSPECIFIED = 0
    DISABLED = 1
    STANDARD = 2
    ENHANCED = 3