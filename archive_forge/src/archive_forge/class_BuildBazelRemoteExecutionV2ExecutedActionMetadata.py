from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildBazelRemoteExecutionV2ExecutedActionMetadata(_messages.Message):
    """ExecutedActionMetadata contains details about a completed execution.

  Messages:
    AuxiliaryMetadataValueListEntry: A AuxiliaryMetadataValueListEntry object.

  Fields:
    auxiliaryMetadata: Details that are specific to the kind of worker used.
      For example, on POSIX-like systems this could contain a message with
      getrusage(2) statistics.
    executionCompletedTimestamp: When the worker completed executing the
      action command.
    executionStartTimestamp: When the worker started executing the action
      command.
    inputFetchCompletedTimestamp: When the worker finished fetching action
      inputs.
    inputFetchStartTimestamp: When the worker started fetching action inputs.
    outputUploadCompletedTimestamp: When the worker finished uploading action
      outputs.
    outputUploadStartTimestamp: When the worker started uploading action
      outputs.
    queuedTimestamp: When was the action added to the queue.
    virtualExecutionDuration: New in v2.3: the amount of time the worker spent
      executing the action command, potentially computed using a worker-
      specific virtual clock. The virtual execution duration is only intended
      to cover the "execution" of the specified action and not time in queue
      nor any overheads before or after execution such as marshalling
      inputs/outputs. The server SHOULD avoid including time spent the client
      doesn't have control over, and MAY extend or reduce the execution
      duration to account for delays or speedups that occur during execution
      itself (e.g., lazily loading data from the Content Addressable Storage,
      live migration of virtual machines, emulation overhead). The method of
      timekeeping used to compute the virtual execution duration MUST be
      consistent with what is used to enforce the Action's `timeout`. There is
      no relationship between the virtual execution duration and the values of
      `execution_start_timestamp` and `execution_completed_timestamp`.
    worker: The name of the worker which ran the execution.
    workerCompletedTimestamp: When the worker completed the action, including
      all stages.
    workerStartTimestamp: When the worker received the action.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AuxiliaryMetadataValueListEntry(_messages.Message):
        """A AuxiliaryMetadataValueListEntry object.

    Messages:
      AdditionalProperty: An additional property for a
        AuxiliaryMetadataValueListEntry object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AuxiliaryMetadataValueListEntry object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    auxiliaryMetadata = _messages.MessageField('AuxiliaryMetadataValueListEntry', 1, repeated=True)
    executionCompletedTimestamp = _messages.StringField(2)
    executionStartTimestamp = _messages.StringField(3)
    inputFetchCompletedTimestamp = _messages.StringField(4)
    inputFetchStartTimestamp = _messages.StringField(5)
    outputUploadCompletedTimestamp = _messages.StringField(6)
    outputUploadStartTimestamp = _messages.StringField(7)
    queuedTimestamp = _messages.StringField(8)
    virtualExecutionDuration = _messages.StringField(9)
    worker = _messages.StringField(10)
    workerCompletedTimestamp = _messages.StringField(11)
    workerStartTimestamp = _messages.StringField(12)