from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectionStatusValueValuesEnum(_messages.Enum):
    """Output only. Connection status.

    Values:
      UNKNOWN: Connection status is unspecified.
      ACCEPTED: Connection is established and functioning normally.
      PENDING: Connection is not established (Looker tenant project hasn't
        been allowlisted).
      REJECTED: Connection is not established (Looker tenant project is
        explicitly in reject list).
      NEEDS_ATTENTION: Issue with target service attachment, e.g. NAT subnet
        is exhausted.
      CLOSED: Target service attachment does not exist. This status is a
        terminal state.
    """
    UNKNOWN = 0
    ACCEPTED = 1
    PENDING = 2
    REJECTED = 3
    NEEDS_ATTENTION = 4
    CLOSED = 5