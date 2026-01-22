from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InvalidReasonValueValuesEnum(_messages.Enum):
    """Output only. Reason associated with the response when valid = false.

    Values:
      INVALID_REASON_UNSPECIFIED: Default unspecified type.
      UNKNOWN_INVALID_REASON: If the failure reason was not accounted for.
      MALFORMED: The provided user verification token was malformed.
      EXPIRED: The user verification token had expired.
      DUPE: The user verification had already been seen.
      MISSING: The user verification token was not present.
      BROWSER_ERROR: A retriable error (such as network failure) occurred on
        the browser. Could easily be simulated by an attacker.
    """
    INVALID_REASON_UNSPECIFIED = 0
    UNKNOWN_INVALID_REASON = 1
    MALFORMED = 2
    EXPIRED = 3
    DUPE = 4
    MISSING = 5
    BROWSER_ERROR = 6