from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CutoverJob(_messages.Message):
    """CutoverJob message describes a cutover of a migrating VM. The CutoverJob
  is the operation of shutting down the VM, creating a snapshot and clonning
  the VM using the replicated snapshot.

  Enums:
    StateValueValuesEnum: Output only. State of the cutover job.

  Fields:
    computeEngineDisksTargetDetails: Output only. Details of the target
      Persistent Disks in Compute Engine.
    computeEngineTargetDetails: Output only. Details of the target VM in
      Compute Engine.
    createTime: Output only. The time the cutover job was created (as an API
      call, not when it was actually created in the target).
    endTime: Output only. The time the cutover job had finished.
    error: Output only. Provides details for the errors that led to the
      Cutover Job's state.
    name: Output only. The name of the cutover job.
    progressPercent: Output only. The current progress in percentage of the
      cutover job.
    state: Output only. State of the cutover job.
    stateMessage: Output only. A message providing possible extra details
      about the current state.
    stateTime: Output only. The time the state was last updated.
    steps: Output only. The cutover steps list representing its progress.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the cutover job.

    Values:
      STATE_UNSPECIFIED: The state is unknown. This is used for API
        compatibility only and is not used by the system.
      PENDING: The cutover job has not yet started.
      FAILED: The cutover job finished with errors.
      SUCCEEDED: The cutover job finished successfully.
      CANCELLED: The cutover job was cancelled.
      CANCELLING: The cutover job is being cancelled.
      ACTIVE: The cutover job is active and running.
      ADAPTING_OS: OS adaptation is running as part of the cutover job to
        generate license.
    """
        STATE_UNSPECIFIED = 0
        PENDING = 1
        FAILED = 2
        SUCCEEDED = 3
        CANCELLED = 4
        CANCELLING = 5
        ACTIVE = 6
        ADAPTING_OS = 7
    computeEngineDisksTargetDetails = _messages.MessageField('ComputeEngineDisksTargetDetails', 1)
    computeEngineTargetDetails = _messages.MessageField('ComputeEngineTargetDetails', 2)
    createTime = _messages.StringField(3)
    endTime = _messages.StringField(4)
    error = _messages.MessageField('Status', 5)
    name = _messages.StringField(6)
    progressPercent = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    state = _messages.EnumField('StateValueValuesEnum', 8)
    stateMessage = _messages.StringField(9)
    stateTime = _messages.StringField(10)
    steps = _messages.MessageField('CutoverStep', 11, repeated=True)