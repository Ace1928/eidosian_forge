from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1beta1Index(_messages.Message):
    """An index definition.

  Enums:
    StateValueValuesEnum: The state of the index. Output only.

  Fields:
    collectionId: The collection ID to which this index applies. Required.
    fields: The fields to index.
    name: The resource name of the index. Output only.
    state: The state of the index. Output only.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The state of the index. Output only.

    Values:
      STATE_UNSPECIFIED: The state is unspecified.
      CREATING: The index is being created. There is an active long-running
        operation for the index. The index is updated when writing a document.
        Some index data may exist.
      READY: The index is ready to be used. The index is updated when writing
        a document. The index is fully populated from all stored documents it
        applies to.
      ERROR: The index was being created, but something went wrong. There is
        no active long-running operation for the index, and the most recently
        finished long-running operation failed. The index is not updated when
        writing a document. Some index data may exist.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        READY = 2
        ERROR = 3
    collectionId = _messages.StringField(1)
    fields = _messages.MessageField('GoogleFirestoreAdminV1beta1IndexField', 2, repeated=True)
    name = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)