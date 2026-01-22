from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ReportPatchJobInstanceDetailsResponse(_messages.Message):
    """Response from reporting instance patch details. Includes information the
  agent needs to continue or stop patching.

  Enums:
    PatchJobStateValueValuesEnum: State of the overall patch. If the patch is
      no longer active, the agent should not begin a new patch step.

  Fields:
    dryRun: If this patch job is a dry run, the agent will report its status
      as it goes through the motions but won't actually run any updates or
      perform any reboots.
    patchConfig: Patch configuration the agent should apply.
    patchJob: Unique identifier for the current patch job.
    patchJobState: State of the overall patch. If the patch is no longer
      active, the agent should not begin a new patch step.
  """

    class PatchJobStateValueValuesEnum(_messages.Enum):
        """State of the overall patch. If the patch is no longer active, the
    agent should not begin a new patch step.

    Values:
      PATCH_JOB_STATE_UNSPECIFIED: Unspecified is invalid.
      ACTIVE: The patch job is running. Instances will continue to run patch
        job steps.
      COMPLETED: The patch job is complete.
    """
        PATCH_JOB_STATE_UNSPECIFIED = 0
        ACTIVE = 1
        COMPLETED = 2
    dryRun = _messages.BooleanField(1)
    patchConfig = _messages.MessageField('PatchConfig', 2)
    patchJob = _messages.StringField(3)
    patchJobState = _messages.EnumField('PatchJobStateValueValuesEnum', 4)