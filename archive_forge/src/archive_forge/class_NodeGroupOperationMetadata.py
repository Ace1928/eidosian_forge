from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeGroupOperationMetadata(_messages.Message):
    """Metadata describing the node group operation.

  Enums:
    OperationTypeValueValuesEnum: The operation type.

  Messages:
    LabelsValue: Output only. Labels associated with the operation.

  Fields:
    clusterUuid: Output only. Cluster UUID associated with the node group
      operation.
    description: Output only. Short description of operation.
    labels: Output only. Labels associated with the operation.
    nodeGroupId: Output only. Node group ID for the operation.
    operationType: The operation type.
    status: Output only. Current operation status.
    statusHistory: Output only. The previous operation status.
    warnings: Output only. Errors encountered during operation execution.
  """

    class OperationTypeValueValuesEnum(_messages.Enum):
        """The operation type.

    Values:
      NODE_GROUP_OPERATION_TYPE_UNSPECIFIED: Node group operation type is
        unknown.
      CREATE: Create node group operation type.
      UPDATE: Update node group operation type.
      DELETE: Delete node group operation type.
      RESIZE: Resize node group operation type.
      REPAIR: Repair node group operation type.
      UPDATE_LABELS: Update node group label operation type.
    """
        NODE_GROUP_OPERATION_TYPE_UNSPECIFIED = 0
        CREATE = 1
        UPDATE = 2
        DELETE = 3
        RESIZE = 4
        REPAIR = 5
        UPDATE_LABELS = 6

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Output only. Labels associated with the operation.

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
    clusterUuid = _messages.StringField(1)
    description = _messages.StringField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    nodeGroupId = _messages.StringField(4)
    operationType = _messages.EnumField('OperationTypeValueValuesEnum', 5)
    status = _messages.MessageField('ClusterOperationStatus', 6)
    statusHistory = _messages.MessageField('ClusterOperationStatus', 7, repeated=True)
    warnings = _messages.StringField(8, repeated=True)