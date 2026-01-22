from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildBazelRemoteExecutionV2ExecuteOperationMetadata(_messages.Message):
    """Metadata about an ongoing execution, which will be contained in the
  metadata field of the Operation.

  Enums:
    StageValueValuesEnum: The current stage of execution.

  Fields:
    actionDigest: The digest of the Action being executed.
    partialExecutionMetadata: The client can read this field to view details
      about the ongoing execution.
    stage: The current stage of execution.
    stderrStreamName: If set, the client can use this resource name with
      ByteStream.Read to stream the standard error from the endpoint hosting
      streamed responses.
    stdoutStreamName: If set, the client can use this resource name with
      ByteStream.Read to stream the standard output from the endpoint hosting
      streamed responses.
  """

    class StageValueValuesEnum(_messages.Enum):
        """The current stage of execution.

    Values:
      UNKNOWN: Invalid value.
      CACHE_CHECK: Checking the result against the cache.
      QUEUED: Currently idle, awaiting a free machine to execute.
      EXECUTING: Currently being executed by a worker.
      COMPLETED: Finished execution.
    """
        UNKNOWN = 0
        CACHE_CHECK = 1
        QUEUED = 2
        EXECUTING = 3
        COMPLETED = 4
    actionDigest = _messages.MessageField('BuildBazelRemoteExecutionV2Digest', 1)
    partialExecutionMetadata = _messages.MessageField('BuildBazelRemoteExecutionV2ExecutedActionMetadata', 2)
    stage = _messages.EnumField('StageValueValuesEnum', 3)
    stderrStreamName = _messages.StringField(4)
    stdoutStreamName = _messages.StringField(5)