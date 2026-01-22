from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MirroringDeploymentGroup(_messages.Message):
    """Message describing MirroringDeploymentGroup object

  Enums:
    StateValueValuesEnum: Output only. Current state of the deployment group.

  Messages:
    LabelsValue: Optional. Labels as key value pairs

  Fields:
    connectedEndpointGroups: Output only. The list of Mirroring Endpoint
      Groups that are connected to this resource.
    createTime: Output only. [Output only] Create time stamp
    labels: Optional. Labels as key value pairs
    name: Immutable. Identifier. Then name of the MirroringDeploymentGroup.
    network: Required. Immutable. The network that is being used for the
      deployment. Format is: projects/{project}/global/networks/{network}.
    reconciling: Output only. Whether reconciling is in progress, recommended
      per https://google.aip.dev/128.
    state: Output only. Current state of the deployment group.
    updateTime: Output only. [Output only] Update time stamp
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Current state of the deployment group.

    Values:
      STATE_UNSPECIFIED: Not set.
      ACTIVE: Ready.
      CREATING: Being created.
      DELETING: Being deleted.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        CREATING = 2
        DELETING = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels as key value pairs

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
    connectedEndpointGroups = _messages.MessageField('MirroringDeploymentGroupConnectedEndpointGroup', 1, repeated=True)
    createTime = _messages.StringField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    name = _messages.StringField(4)
    network = _messages.StringField(5)
    reconciling = _messages.BooleanField(6)
    state = _messages.EnumField('StateValueValuesEnum', 7)
    updateTime = _messages.StringField(8)