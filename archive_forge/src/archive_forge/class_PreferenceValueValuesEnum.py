from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PreferenceValueValuesEnum(_messages.Enum):
    """Preference for a given location. Set to either ALLOW or DENY.

    Values:
      ALLOW: Location is allowed for use.
      DENY: Location is prohibited.
      PREFERENCE_UNSPECIFIED: Default value, unused.
    """
    ALLOW = 0
    DENY = 1
    PREFERENCE_UNSPECIFIED = 2