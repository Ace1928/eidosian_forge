from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SessionOperationMetadata(_messages.Message):
    """Metadata describing the Session operation.

  Enums:
    OperationTypeValueValuesEnum: The operation type.

  Messages:
    LabelsValue: Labels associated with the operation.

  Fields:
    createTime: The time when the operation was created.
    description: Short description of the operation.
    doneTime: The time when the operation was finished.
    labels: Labels associated with the operation.
    operationType: The operation type.
    session: Name of the session for the operation.
    sessionUuid: Session UUID for the operation.
    warnings: Warnings encountered during operation execution.
  """

    class OperationTypeValueValuesEnum(_messages.Enum):
        """The operation type.

    Values:
      SESSION_OPERATION_TYPE_UNSPECIFIED: Session operation type is unknown.
      CREATE: Create Session operation type.
      TERMINATE: Terminate Session operation type.
      DELETE: Delete Session operation type.
    """
        SESSION_OPERATION_TYPE_UNSPECIFIED = 0
        CREATE = 1
        TERMINATE = 2
        DELETE = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels associated with the operation.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    doneTime = _messages.StringField(3)
    labels = _messages.MessageField('LabelsValue', 4)
    operationType = _messages.EnumField('OperationTypeValueValuesEnum', 5)
    session = _messages.StringField(6)
    sessionUuid = _messages.StringField(7)
    warnings = _messages.StringField(8, repeated=True)