from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectionPreferenceValueValuesEnum(_messages.Enum):
    """The connection preference of service attachment. The value can be set
    to ACCEPT_AUTOMATIC. An ACCEPT_AUTOMATIC service attachment is one that
    always accepts the connection from consumer forwarding rules.

    Values:
      ACCEPT_AUTOMATIC: <no description>
      ACCEPT_MANUAL: <no description>
      CONNECTION_PREFERENCE_UNSPECIFIED: <no description>
    """
    ACCEPT_AUTOMATIC = 0
    ACCEPT_MANUAL = 1
    CONNECTION_PREFERENCE_UNSPECIFIED = 2