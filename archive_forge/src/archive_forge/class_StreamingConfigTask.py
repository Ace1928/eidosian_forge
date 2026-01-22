from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StreamingConfigTask(_messages.Message):
    """A task that carries configuration information for streaming
  computations.

  Messages:
    UserStepToStateFamilyNameMapValue: Map from user step names to state
      families.

  Fields:
    commitStreamChunkSizeBytes: Chunk size for commit streams from the harness
      to windmill.
    getDataStreamChunkSizeBytes: Chunk size for get data streams from the
      harness to windmill.
    maxWorkItemCommitBytes: Maximum size for work item commit supported
      windmill storage layer.
    streamingComputationConfigs: Set of computation configuration information.
    userStepToStateFamilyNameMap: Map from user step names to state families.
    windmillServiceEndpoint: If present, the worker must use this endpoint to
      communicate with Windmill Service dispatchers, otherwise the worker must
      continue to use whatever endpoint it had been using.
    windmillServicePort: If present, the worker must use this port to
      communicate with Windmill Service dispatchers. Only applicable when
      windmill_service_endpoint is specified.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class UserStepToStateFamilyNameMapValue(_messages.Message):
        """Map from user step names to state families.

    Messages:
      AdditionalProperty: An additional property for a
        UserStepToStateFamilyNameMapValue object.

    Fields:
      additionalProperties: Additional properties of type
        UserStepToStateFamilyNameMapValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a UserStepToStateFamilyNameMapValue
      object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    commitStreamChunkSizeBytes = _messages.IntegerField(1)
    getDataStreamChunkSizeBytes = _messages.IntegerField(2)
    maxWorkItemCommitBytes = _messages.IntegerField(3)
    streamingComputationConfigs = _messages.MessageField('StreamingComputationConfig', 4, repeated=True)
    userStepToStateFamilyNameMap = _messages.MessageField('UserStepToStateFamilyNameMapValue', 5)
    windmillServiceEndpoint = _messages.StringField(6)
    windmillServicePort = _messages.IntegerField(7)