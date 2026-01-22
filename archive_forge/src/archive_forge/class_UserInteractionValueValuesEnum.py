from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserInteractionValueValuesEnum(_messages.Enum):
    """This metric captures the requirement for a human user, other than the
    attacker, to participate in the successful compromise of the vulnerable
    component.

    Values:
      USER_INTERACTION_UNSPECIFIED: Invalid value.
      USER_INTERACTION_NONE: The vulnerable system can be exploited without
        interaction from any user.
      USER_INTERACTION_REQUIRED: Successful exploitation of this vulnerability
        requires a user to take some action before the vulnerability can be
        exploited.
    """
    USER_INTERACTION_UNSPECIFIED = 0
    USER_INTERACTION_NONE = 1
    USER_INTERACTION_REQUIRED = 2