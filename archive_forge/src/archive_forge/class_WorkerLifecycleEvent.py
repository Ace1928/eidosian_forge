from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkerLifecycleEvent(_messages.Message):
    """A report of an event in a worker's lifecycle. The proto contains one
  event, because the worker is expected to asynchronously send each message
  immediately after the event. Due to this asynchrony, messages may arrive out
  of order (or missing), and it is up to the consumer to interpret. The
  timestamp of the event is in the enclosing WorkerMessage proto.

  Enums:
    EventValueValuesEnum: The event being reported.

  Messages:
    MetadataValue: Other stats that can accompany an event. E.g. {
      "downloaded_bytes" : "123456" }

  Fields:
    containerStartTime: The start time of this container. All events will
      report this so that events can be grouped together across container/VM
      restarts.
    event: The event being reported.
    metadata: Other stats that can accompany an event. E.g. {
      "downloaded_bytes" : "123456" }
  """

    class EventValueValuesEnum(_messages.Enum):
        """The event being reported.

    Values:
      UNKNOWN_EVENT: Invalid event.
      OS_START: The time the VM started.
      CONTAINER_START: Our container code starts running. Multiple containers
        could be distinguished with WorkerMessage.labels if desired.
      NETWORK_UP: The worker has a functional external network connection.
      STAGING_FILES_DOWNLOAD_START: Started downloading staging files.
      STAGING_FILES_DOWNLOAD_FINISH: Finished downloading all staging files.
      SDK_INSTALL_START: For applicable SDKs, started installation of SDK and
        worker packages.
      SDK_INSTALL_FINISH: Finished installing SDK.
    """
        UNKNOWN_EVENT = 0
        OS_START = 1
        CONTAINER_START = 2
        NETWORK_UP = 3
        STAGING_FILES_DOWNLOAD_START = 4
        STAGING_FILES_DOWNLOAD_FINISH = 5
        SDK_INSTALL_START = 6
        SDK_INSTALL_FINISH = 7

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetadataValue(_messages.Message):
        """Other stats that can accompany an event. E.g. { "downloaded_bytes" :
    "123456" }

    Messages:
      AdditionalProperty: An additional property for a MetadataValue object.

    Fields:
      additionalProperties: Additional properties of type MetadataValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    containerStartTime = _messages.StringField(1)
    event = _messages.EnumField('EventValueValuesEnum', 2)
    metadata = _messages.MessageField('MetadataValue', 3)