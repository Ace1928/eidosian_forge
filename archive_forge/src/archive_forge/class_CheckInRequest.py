from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CheckInRequest(_messages.Message):
    """The parameters to the CheckIn method.

  Messages:
    EventValue: A workflow specific event occurred.

  Fields:
    deadlineExpired: The deadline has expired and the worker needs more time.
    event: A workflow specific event occurred.
    events: A list of timestamped events.
    result: The operation has finished with the given result.
    sosReport: An SOS report for an unexpected VM failure.
    workerStatus: Data about the status of the worker VM.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class EventValue(_messages.Message):
        """A workflow specific event occurred.

    Messages:
      AdditionalProperty: An additional property for a EventValue object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a EventValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    deadlineExpired = _messages.MessageField('Empty', 1)
    event = _messages.MessageField('EventValue', 2)
    events = _messages.MessageField('TimestampedEvent', 3, repeated=True)
    result = _messages.MessageField('Status', 4)
    sosReport = _messages.BytesField(5)
    workerStatus = _messages.MessageField('WorkerStatus', 6)