from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1OperationMetadata(_messages.Message):
    """Represents the metadata of the long-running operation.

  Enums:
    OperationTypeValueValuesEnum: The operation type.

  Messages:
    LabelsValue: The user labels, inherited from the model or the model
      version being operated on.

  Fields:
    createTime: The time the operation was submitted.
    endTime: The time operation processing completed.
    isCancellationRequested: Indicates whether a request to cancel this
      operation has been made.
    labels: The user labels, inherited from the model or the model version
      being operated on.
    modelName: Contains the name of the model associated with the operation.
    operationType: The operation type.
    projectNumber: Contains the project number associated with the operation.
    startTime: The time operation processing started.
    version: Contains the version associated with the operation.
  """

    class OperationTypeValueValuesEnum(_messages.Enum):
        """The operation type.

    Values:
      OPERATION_TYPE_UNSPECIFIED: Unspecified operation type.
      CREATE_VERSION: An operation to create a new version.
      DELETE_VERSION: An operation to delete an existing version.
      DELETE_MODEL: An operation to delete an existing model.
      UPDATE_MODEL: An operation to update an existing model.
      UPDATE_VERSION: An operation to update an existing version.
      UPDATE_CONFIG: An operation to update project configuration.
    """
        OPERATION_TYPE_UNSPECIFIED = 0
        CREATE_VERSION = 1
        DELETE_VERSION = 2
        DELETE_MODEL = 3
        UPDATE_MODEL = 4
        UPDATE_VERSION = 5
        UPDATE_CONFIG = 6

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The user labels, inherited from the model or the model version being
    operated on.

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
    endTime = _messages.StringField(2)
    isCancellationRequested = _messages.BooleanField(3)
    labels = _messages.MessageField('LabelsValue', 4)
    modelName = _messages.StringField(5)
    operationType = _messages.EnumField('OperationTypeValueValuesEnum', 6)
    projectNumber = _messages.IntegerField(7)
    startTime = _messages.StringField(8)
    version = _messages.MessageField('GoogleCloudMlV1Version', 9)