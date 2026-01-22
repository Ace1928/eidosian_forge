from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1SessionEvent(_messages.Message):
    """These messages contain information about sessions within an environment.
  The monitored resource is 'Environment'.

  Enums:
    TypeValueValuesEnum: The type of the event.

  Fields:
    eventSucceeded: The status of the event.
    fastStartupEnabled: If the session is associated with an environment with
      fast startup enabled, and was created before being assigned to a user.
    message: The log message.
    query: The execution details of the query.
    sessionId: Unique identifier for the session.
    type: The type of the event.
    unassignedDuration: The idle duration of a warm pooled session before it
      is assigned to user.
    userId: The information about the user that created the session. It will
      be the email address of the user.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of the event.

    Values:
      EVENT_TYPE_UNSPECIFIED: An unspecified event type.
      START: Event when the session is assigned to a user.
      STOP: Event for stop of a session.
      QUERY: Query events in the session.
      CREATE: Event for creation of a cluster. It is not yet assigned to a
        user. This comes before START in the sequence
    """
        EVENT_TYPE_UNSPECIFIED = 0
        START = 1
        STOP = 2
        QUERY = 3
        CREATE = 4
    eventSucceeded = _messages.BooleanField(1)
    fastStartupEnabled = _messages.BooleanField(2)
    message = _messages.StringField(3)
    query = _messages.MessageField('GoogleCloudDataplexV1SessionEventQueryDetail', 4)
    sessionId = _messages.StringField(5)
    type = _messages.EnumField('TypeValueValuesEnum', 6)
    unassignedDuration = _messages.StringField(7)
    userId = _messages.StringField(8)