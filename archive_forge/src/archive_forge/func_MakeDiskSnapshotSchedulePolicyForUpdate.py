from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.resource_policies import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
def MakeDiskSnapshotSchedulePolicyForUpdate(policy_ref, args, messages):
    """Creates a Disk Snapshot Schedule Resource Policy message from args used in ResourcePolicy.Patch.
  """
    hourly_cycle, daily_cycle, weekly_cycle = _ParseCycleFrequencyArgs(args, messages, supports_hourly=True, supports_weekly=True)
    snapshot_properties, snapshot_schedule, description = (None, None, None)
    snapshot_labels = labels_util.ParseCreateArgs(args, messages.ResourcePolicySnapshotSchedulePolicySnapshotProperties.LabelsValue, labels_dest='snapshot_labels')
    if snapshot_labels:
        snapshot_properties = messages.ResourcePolicySnapshotSchedulePolicySnapshotProperties(labels=snapshot_labels)
    if args.IsSpecified('description'):
        description = args.description
    retention_policy = None
    if args.max_retention_days or args.on_source_disk_delete:
        retention_policy = messages.ResourcePolicySnapshotSchedulePolicyRetentionPolicy(maxRetentionDays=args.max_retention_days, onSourceDiskDelete=flags.GetOnSourceDiskDeleteFlagMapper(messages).GetEnumForChoice(args.on_source_disk_delete))
    if hourly_cycle or daily_cycle or weekly_cycle:
        snapshot_schedule = messages.ResourcePolicySnapshotSchedulePolicySchedule(hourlySchedule=hourly_cycle, dailySchedule=daily_cycle, weeklySchedule=weekly_cycle)
    snapshot_policy = None
    if snapshot_schedule or snapshot_properties or retention_policy:
        snapshot_policy = messages.ResourcePolicySnapshotSchedulePolicy(schedule=snapshot_schedule, snapshotProperties=snapshot_properties, retentionPolicy=retention_policy)
    return messages.ResourcePolicy(name=policy_ref.Name(), description=description, snapshotSchedulePolicy=snapshot_policy)