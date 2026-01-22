from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PasswordProtectionWarningTriggerValueValuesEnum(_messages.Enum):
    """Current state of [password protection trigger](https://chromeenterpris
    e.google/policies/#PasswordProtectionWarningTrigger).

    Values:
      PASSWORD_PROTECTION_TRIGGER_UNSPECIFIED: Password protection is not
        specified.
      PROTECTION_OFF: Password reuse is never detected.
      PASSWORD_REUSE: Warning is shown when the user reuses their protected
        password on a non-allowed site.
      PHISHING_REUSE: Warning is shown when the user reuses their protected
        password on a phishing site.
    """
    PASSWORD_PROTECTION_TRIGGER_UNSPECIFIED = 0
    PROTECTION_OFF = 1
    PASSWORD_REUSE = 2
    PHISHING_REUSE = 3