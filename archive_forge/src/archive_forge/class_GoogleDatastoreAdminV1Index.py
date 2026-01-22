from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDatastoreAdminV1Index(_messages.Message):
    """Datastore composite index definition.

  Enums:
    AncestorValueValuesEnum: Required. The index's ancestor mode. Must not be
      ANCESTOR_MODE_UNSPECIFIED.
    StateValueValuesEnum: Output only. The state of the index.

  Fields:
    ancestor: Required. The index's ancestor mode. Must not be
      ANCESTOR_MODE_UNSPECIFIED.
    indexId: Output only. The resource ID of the index.
    kind: Required. The entity kind to which this index applies.
    projectId: Output only. Project ID.
    properties: Required. An ordered sequence of property names and their
      index attributes. Requires: * A maximum of 100 properties.
    state: Output only. The state of the index.
  """

    class AncestorValueValuesEnum(_messages.Enum):
        """Required. The index's ancestor mode. Must not be
    ANCESTOR_MODE_UNSPECIFIED.

    Values:
      ANCESTOR_MODE_UNSPECIFIED: The ancestor mode is unspecified.
      NONE: Do not include the entity's ancestors in the index.
      ALL_ANCESTORS: Include all the entity's ancestors in the index.
    """
        ANCESTOR_MODE_UNSPECIFIED = 0
        NONE = 1
        ALL_ANCESTORS = 2

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the index.

    Values:
      STATE_UNSPECIFIED: The state is unspecified.
      CREATING: The index is being created, and cannot be used by queries.
        There is an active long-running operation for the index. The index is
        updated when writing an entity. Some index data may exist.
      READY: The index is ready to be used. The index is updated when writing
        an entity. The index is fully populated from all stored entities it
        applies to.
      DELETING: The index is being deleted, and cannot be used by queries.
        There is an active long-running operation for the index. The index is
        not updated when writing an entity. Some index data may exist.
      ERROR: The index was being created or deleted, but something went wrong.
        The index cannot by used by queries. There is no active long-running
        operation for the index, and the most recently finished long-running
        operation failed. The index is not updated when writing an entity.
        Some index data may exist.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        READY = 2
        DELETING = 3
        ERROR = 4
    ancestor = _messages.EnumField('AncestorValueValuesEnum', 1)
    indexId = _messages.StringField(2)
    kind = _messages.StringField(3)
    projectId = _messages.StringField(4)
    properties = _messages.MessageField('GoogleDatastoreAdminV1IndexedProperty', 5, repeated=True)
    state = _messages.EnumField('StateValueValuesEnum', 6)