from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ParticipantRoleValueValuesEnum(_messages.Enum):
    """Required. The participant role to add or update the suggestion feature
    config. Only HUMAN_AGENT or END_USER can be used.

    Values:
      ROLE_UNSPECIFIED: Participant role not set.
      HUMAN_AGENT: Participant is a human agent.
      AUTOMATED_AGENT: Participant is an automated agent, such as a Dialogflow
        agent.
      END_USER: Participant is an end user that has called or chatted with
        Dialogflow services.
    """
    ROLE_UNSPECIFIED = 0
    HUMAN_AGENT = 1
    AUTOMATED_AGENT = 2
    END_USER = 3