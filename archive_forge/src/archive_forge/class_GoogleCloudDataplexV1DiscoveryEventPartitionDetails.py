from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DiscoveryEventPartitionDetails(_messages.Message):
    """Details about the partition.

  Enums:
    TypeValueValuesEnum: The type of the containing entity resource.

  Fields:
    entity: The name to the containing entity resource. The name is the fully-
      qualified resource name.
    partition: The name to the partition resource. The name is the fully-
      qualified resource name.
    sampledDataLocations: The locations of the data items (e.g., a Cloud
      Storage objects) sampled for metadata inference.
    type: The type of the containing entity resource.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of the containing entity resource.

    Values:
      ENTITY_TYPE_UNSPECIFIED: An unspecified event type.
      TABLE: Entities representing structured data.
      FILESET: Entities representing unstructured data.
    """
        ENTITY_TYPE_UNSPECIFIED = 0
        TABLE = 1
        FILESET = 2
    entity = _messages.StringField(1)
    partition = _messages.StringField(2)
    sampledDataLocations = _messages.StringField(3, repeated=True)
    type = _messages.EnumField('TypeValueValuesEnum', 4)