from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DiscoveryEvent(_messages.Message):
    """The payload associated with Discovery data processing.

  Enums:
    TypeValueValuesEnum: The type of the event being logged.

  Fields:
    action: Details about the action associated with the event.
    assetId: The id of the associated asset.
    config: Details about discovery configuration in effect.
    dataLocation: The data location associated with the event.
    entity: Details about the entity associated with the event.
    lakeId: The id of the associated lake.
    message: The log message.
    partition: Details about the partition associated with the event.
    type: The type of the event being logged.
    zoneId: The id of the associated zone.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of the event being logged.

    Values:
      EVENT_TYPE_UNSPECIFIED: An unspecified event type.
      CONFIG: An event representing discovery configuration in effect.
      ENTITY_CREATED: An event representing a metadata entity being created.
      ENTITY_UPDATED: An event representing a metadata entity being updated.
      ENTITY_DELETED: An event representing a metadata entity being deleted.
      PARTITION_CREATED: An event representing a partition being created.
      PARTITION_UPDATED: An event representing a partition being updated.
      PARTITION_DELETED: An event representing a partition being deleted.
    """
        EVENT_TYPE_UNSPECIFIED = 0
        CONFIG = 1
        ENTITY_CREATED = 2
        ENTITY_UPDATED = 3
        ENTITY_DELETED = 4
        PARTITION_CREATED = 5
        PARTITION_UPDATED = 6
        PARTITION_DELETED = 7
    action = _messages.MessageField('GoogleCloudDataplexV1DiscoveryEventActionDetails', 1)
    assetId = _messages.StringField(2)
    config = _messages.MessageField('GoogleCloudDataplexV1DiscoveryEventConfigDetails', 3)
    dataLocation = _messages.StringField(4)
    entity = _messages.MessageField('GoogleCloudDataplexV1DiscoveryEventEntityDetails', 5)
    lakeId = _messages.StringField(6)
    message = _messages.StringField(7)
    partition = _messages.MessageField('GoogleCloudDataplexV1DiscoveryEventPartitionDetails', 8)
    type = _messages.EnumField('TypeValueValuesEnum', 9)
    zoneId = _messages.StringField(10)