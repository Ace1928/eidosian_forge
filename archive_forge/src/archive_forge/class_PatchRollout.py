from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PatchRollout(_messages.Message):
    """Patch rollout configuration specifications. Contains details on the
  concurrency control when applying patch(es) to all targeted VMs.

  Enums:
    ModeValueValuesEnum: Mode of the patch rollout.

  Fields:
    disruptionBudget: The maximum number (or percentage) of VMs per zone to
      disrupt at any given moment. The number of VMs calculated from
      multiplying the percentage by the total number of VMs in a zone is
      rounded up. During patching, a VM is considered disrupted from the time
      the agent is notified to begin until patching has completed. This
      disruption time includes the time to complete reboot and any post-patch
      steps. A VM contributes to the disruption budget if its patching
      operation fails either when applying the patches, running pre or post
      patch steps, or if it fails to respond with a success notification
      before timing out. VMs that are not running or do not have an active
      agent do not count toward this disruption budget. For zone-by-zone
      rollouts, if the disruption budget in a zone is exceeded, the patch job
      stops, because continuing to the next zone requires completion of the
      patch process in the previous zone. For example, if the disruption
      budget has a fixed value of `10`, and 8 VMs fail to patch in the current
      zone, the patch job continues to patch 2 VMs at a time until the zone is
      completed. When that zone is completed successfully, patching begins
      with 10 VMs at a time in the next zone. If 10 VMs in the next zone fail
      to patch, the patch job stops.
    mode: Mode of the patch rollout.
  """

    class ModeValueValuesEnum(_messages.Enum):
        """Mode of the patch rollout.

    Values:
      MODE_UNSPECIFIED: Mode must be specified.
      ZONE_BY_ZONE: Patches are applied one zone at a time. The patch job
        begins in the region with the lowest number of targeted VMs. Within
        the region, patching begins in the zone with the lowest number of
        targeted VMs. If multiple regions (or zones within a region) have the
        same number of targeted VMs, a tie-breaker is achieved by sorting the
        regions or zones in alphabetical order.
      CONCURRENT_ZONES: Patches are applied to VMs in all zones at the same
        time.
    """
        MODE_UNSPECIFIED = 0
        ZONE_BY_ZONE = 1
        CONCURRENT_ZONES = 2
    disruptionBudget = _messages.MessageField('FixedOrPercent', 1)
    mode = _messages.EnumField('ModeValueValuesEnum', 2)