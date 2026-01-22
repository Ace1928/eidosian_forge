from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MirroringEndpointGroupAssociation(_messages.Message):
    """Message describing MirroringEndpointGroupAssociation object

  Enums:
    StateValueValuesEnum: Output only. Current state of the endpoint group
      association.

  Messages:
    LabelsValue: Optional. Labels as key value pairs

  Fields:
    createTime: Output only. [Output only] Create time stamp
    labels: Optional. Labels as key value pairs
    locationsDetails: Output only. The list of locations that this association
      is in and its details.
    mirroringEndpointGroup: Required. Immutable. The Mirroring Endpoint Group
      that this resource is connected to. Format is: `organizations/{organizat
      ion}/locations/global/mirroringEndpointGroups/{mirroringEndpointGroup}`
    name: Immutable. Identifier. The name of the
      MirroringEndpointGroupAssociation.
    network: Required. Immutable. The VPC network associated. Format:
      projects/{project}/global/networks/{network}.
    reconciling: Output only. Whether reconciling is in progress, recommended
      per https://google.aip.dev/128.
    state: Output only. Current state of the endpoint group association.
    updateTime: Output only. [Output only] Update time stamp
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Current state of the endpoint group association.

    Values:
      STATE_UNSPECIFIED: Not set.
      ACTIVE: Ready.
      INACTIVE: The resource is partially not ready, some associations might
        not be in a valid state.
      CREATING: Being created.
      DELETING: Being deleted.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        INACTIVE = 2
        CREATING = 3
        DELETING = 4

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
    createTime = _messages.StringField(1)
    labels = _messages.MessageField('LabelsValue', 2)
    locationsDetails = _messages.MessageField('MirroringEndpointGroupAssociationLocationDetails', 3, repeated=True)
    mirroringEndpointGroup = _messages.StringField(4)
    name = _messages.StringField(5)
    network = _messages.StringField(6)
    reconciling = _messages.BooleanField(7)
    state = _messages.EnumField('StateValueValuesEnum', 8)
    updateTime = _messages.StringField(9)