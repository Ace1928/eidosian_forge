from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TopologyConfig(_messages.Message):
    """Global topology of the streaming Dataflow job, including all
  computations and their sharded locations.

  Messages:
    UserStageToComputationNameMapValue: Maps user stage names to stable
      computation names.

  Fields:
    computations: The computations associated with a streaming Dataflow job.
    dataDiskAssignments: The disks assigned to a streaming Dataflow job.
    forwardingKeyBits: The size (in bits) of keys that will be assigned to
      source messages.
    persistentStateVersion: Version number for persistent state.
    userStageToComputationNameMap: Maps user stage names to stable computation
      names.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class UserStageToComputationNameMapValue(_messages.Message):
        """Maps user stage names to stable computation names.

    Messages:
      AdditionalProperty: An additional property for a
        UserStageToComputationNameMapValue object.

    Fields:
      additionalProperties: Additional properties of type
        UserStageToComputationNameMapValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a UserStageToComputationNameMapValue
      object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    computations = _messages.MessageField('ComputationTopology', 1, repeated=True)
    dataDiskAssignments = _messages.MessageField('DataDiskAssignment', 2, repeated=True)
    forwardingKeyBits = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    persistentStateVersion = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    userStageToComputationNameMap = _messages.MessageField('UserStageToComputationNameMapValue', 5)