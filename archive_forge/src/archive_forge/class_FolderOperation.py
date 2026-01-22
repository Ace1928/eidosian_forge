from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FolderOperation(_messages.Message):
    """Metadata describing a long running folder operation

  Enums:
    OperationTypeValueValuesEnum: The type of this operation.

  Fields:
    destinationParent: The resource name of the folder or organization we are
      either creating the folder under or moving the folder to.
    displayName: The display name of the folder.
    operationType: The type of this operation.
    sourceParent: The resource name of the folder's parent. Only applicable
      when the operation_type is MOVE.
  """

    class OperationTypeValueValuesEnum(_messages.Enum):
        """The type of this operation.

    Values:
      OPERATION_TYPE_UNSPECIFIED: Operation type not specified.
      CREATE: A create folder operation.
      MOVE: A move folder operation.
    """
        OPERATION_TYPE_UNSPECIFIED = 0
        CREATE = 1
        MOVE = 2
    destinationParent = _messages.StringField(1)
    displayName = _messages.StringField(2)
    operationType = _messages.EnumField('OperationTypeValueValuesEnum', 3)
    sourceParent = _messages.StringField(4)