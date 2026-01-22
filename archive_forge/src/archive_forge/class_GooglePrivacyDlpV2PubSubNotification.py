from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2PubSubNotification(_messages.Message):
    """Send a Pub/Sub message into the given Pub/Sub topic to connect other
  systems to data profile generation. The message payload data will be the
  byte serialization of `DataProfilePubSubMessage`.

  Enums:
    DetailOfMessageValueValuesEnum: How much data to include in the Pub/Sub
      message. If the user wishes to limit the size of the message, they can
      use resource_name and fetch the profile fields they wish to. Per table
      profile (not per column).
    EventValueValuesEnum: The type of event that triggers a Pub/Sub. At most
      one `PubSubNotification` per EventType is permitted.

  Fields:
    detailOfMessage: How much data to include in the Pub/Sub message. If the
      user wishes to limit the size of the message, they can use resource_name
      and fetch the profile fields they wish to. Per table profile (not per
      column).
    event: The type of event that triggers a Pub/Sub. At most one
      `PubSubNotification` per EventType is permitted.
    pubsubCondition: Conditions (e.g., data risk or sensitivity level) for
      triggering a Pub/Sub.
    topic: Cloud Pub/Sub topic to send notifications to. Format is
      projects/{project}/topics/{topic}.
  """

    class DetailOfMessageValueValuesEnum(_messages.Enum):
        """How much data to include in the Pub/Sub message. If the user wishes to
    limit the size of the message, they can use resource_name and fetch the
    profile fields they wish to. Per table profile (not per column).

    Values:
      DETAIL_LEVEL_UNSPECIFIED: Unused.
      TABLE_PROFILE: The full table data profile.
      RESOURCE_NAME: The resource name of the table.
    """
        DETAIL_LEVEL_UNSPECIFIED = 0
        TABLE_PROFILE = 1
        RESOURCE_NAME = 2

    class EventValueValuesEnum(_messages.Enum):
        """The type of event that triggers a Pub/Sub. At most one
    `PubSubNotification` per EventType is permitted.

    Values:
      EVENT_TYPE_UNSPECIFIED: Unused.
      NEW_PROFILE: New profile (not a re-profile).
      CHANGED_PROFILE: Changed one of the following profile metrics: * Table
        data risk score * Table sensitivity score * Table resource visibility
        * Table encryption type * Table predicted infoTypes * Table other
        infoTypes
      SCORE_INCREASED: Table data risk score or sensitivity score increased.
      ERROR_CHANGED: A user (non-internal) error occurred.
    """
        EVENT_TYPE_UNSPECIFIED = 0
        NEW_PROFILE = 1
        CHANGED_PROFILE = 2
        SCORE_INCREASED = 3
        ERROR_CHANGED = 4
    detailOfMessage = _messages.EnumField('DetailOfMessageValueValuesEnum', 1)
    event = _messages.EnumField('EventValueValuesEnum', 2)
    pubsubCondition = _messages.MessageField('GooglePrivacyDlpV2DataProfilePubSubCondition', 3)
    topic = _messages.StringField(4)