from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PatchJobInstanceDetailsSummary(_messages.Message):
    """A summary of the current patch state across all instances that this
  patch job affects. Contains counts of instances in different states. These
  states map to `InstancePatchState`. List patch job instance details to see
  the specific states of each instance.

  Fields:
    ackedInstanceCount: Number of instances that have acked and will start
      shortly.
    applyingPatchesInstanceCount: Number of instances that are applying
      patches.
    downloadingPatchesInstanceCount: Number of instances that are downloading
      patches.
    failedInstanceCount: Number of instances that failed.
    inactiveInstanceCount: Number of instances that are inactive.
    noAgentDetectedInstanceCount: Number of instances that do not appear to be
      running the agent. Check to ensure that the agent is installed, running,
      and able to communicate with the service.
    notifiedInstanceCount: Number of instances notified about patch job.
    pendingInstanceCount: Number of instances pending patch job.
    postPatchStepInstanceCount: Number of instances that are running the post-
      patch step.
    prePatchStepInstanceCount: Number of instances that are running the pre-
      patch step.
    rebootingInstanceCount: Number of instances rebooting.
    startedInstanceCount: Number of instances that have started.
    succeededInstanceCount: Number of instances that have completed
      successfully.
    succeededRebootRequiredInstanceCount: Number of instances that require
      reboot.
    timedOutInstanceCount: Number of instances that exceeded the time out
      while applying the patch.
  """
    ackedInstanceCount = _messages.IntegerField(1)
    applyingPatchesInstanceCount = _messages.IntegerField(2)
    downloadingPatchesInstanceCount = _messages.IntegerField(3)
    failedInstanceCount = _messages.IntegerField(4)
    inactiveInstanceCount = _messages.IntegerField(5)
    noAgentDetectedInstanceCount = _messages.IntegerField(6)
    notifiedInstanceCount = _messages.IntegerField(7)
    pendingInstanceCount = _messages.IntegerField(8)
    postPatchStepInstanceCount = _messages.IntegerField(9)
    prePatchStepInstanceCount = _messages.IntegerField(10)
    rebootingInstanceCount = _messages.IntegerField(11)
    startedInstanceCount = _messages.IntegerField(12)
    succeededInstanceCount = _messages.IntegerField(13)
    succeededRebootRequiredInstanceCount = _messages.IntegerField(14)
    timedOutInstanceCount = _messages.IntegerField(15)