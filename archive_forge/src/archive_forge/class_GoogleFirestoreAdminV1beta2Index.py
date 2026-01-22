from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1beta2Index(_messages.Message):
    """Cloud Firestore indexes enable simple and complex queries against
  documents in a database.

  Enums:
    QueryScopeValueValuesEnum: Indexes with a collection query scope specified
      allow queries against a collection that is the child of a specific
      document, specified at query time, and that has the same collection id.
      Indexes with a collection group query scope specified allow queries
      against all collections descended from a specific document, specified at
      query time, and that have the same collection id as this index.
    StateValueValuesEnum: Output only. The serving state of the index.

  Fields:
    fields: The fields supported by this index. For composite indexes, this is
      always 2 or more fields. The last field entry is always for the field
      path `__name__`. If, on creation, `__name__` was not specified as the
      last field, it will be added automatically with the same direction as
      that of the last field defined. If the final field in a composite index
      is not directional, the `__name__` will be ordered ASCENDING (unless
      explicitly specified). For single field indexes, this will always be
      exactly one entry with a field path equal to the field path of the
      associated field.
    name: Output only. A server defined name for this index. The form of this
      name for composite indexes will be: `projects/{project_id}/databases/{da
      tabase_id}/collectionGroups/{collection_id}/indexes/{composite_index_id}
      ` For single field indexes, this field will be empty.
    queryScope: Indexes with a collection query scope specified allow queries
      against a collection that is the child of a specific document, specified
      at query time, and that has the same collection id. Indexes with a
      collection group query scope specified allow queries against all
      collections descended from a specific document, specified at query time,
      and that have the same collection id as this index.
    state: Output only. The serving state of the index.
  """

    class QueryScopeValueValuesEnum(_messages.Enum):
        """Indexes with a collection query scope specified allow queries against
    a collection that is the child of a specific document, specified at query
    time, and that has the same collection id. Indexes with a collection group
    query scope specified allow queries against all collections descended from
    a specific document, specified at query time, and that have the same
    collection id as this index.

    Values:
      QUERY_SCOPE_UNSPECIFIED: The query scope is unspecified. Not a valid
        option.
      COLLECTION: Indexes with a collection query scope specified allow
        queries against a collection that is the child of a specific document,
        specified at query time, and that has the collection id specified by
        the index.
      COLLECTION_GROUP: Indexes with a collection group query scope specified
        allow queries against all collections that has the collection id
        specified by the index.
    """
        QUERY_SCOPE_UNSPECIFIED = 0
        COLLECTION = 1
        COLLECTION_GROUP = 2

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The serving state of the index.

    Values:
      STATE_UNSPECIFIED: The state is unspecified.
      CREATING: The index is being created. There is an active long-running
        operation for the index. The index is updated when writing a document.
        Some index data may exist.
      READY: The index is ready to be used. The index is updated when writing
        a document. The index is fully populated from all stored documents it
        applies to.
      NEEDS_REPAIR: The index was being created, but something went wrong.
        There is no active long-running operation for the index, and the most
        recently finished long-running operation failed. The index is not
        updated when writing a document. Some index data may exist. Use the
        google.longrunning.Operations API to determine why the operation that
        last attempted to create this index failed, then re-create the index.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        READY = 2
        NEEDS_REPAIR = 3
    fields = _messages.MessageField('GoogleFirestoreAdminV1beta2IndexField', 1, repeated=True)
    name = _messages.StringField(2)
    queryScope = _messages.EnumField('QueryScopeValueValuesEnum', 3)
    state = _messages.EnumField('StateValueValuesEnum', 4)