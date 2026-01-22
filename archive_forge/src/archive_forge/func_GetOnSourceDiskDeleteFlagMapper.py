from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
def GetOnSourceDiskDeleteFlagMapper(messages):
    return arg_utils.ChoiceEnumMapper('--on-source-disk-delete', messages.ResourcePolicySnapshotSchedulePolicyRetentionPolicy.OnSourceDiskDeleteValueValuesEnum, custom_mappings={'KEEP_AUTO_SNAPSHOTS': ('keep-auto-snapshots', 'Keep automatically-created snapshots when the source disk is deleted. This is the default behavior.'), 'APPLY_RETENTION_POLICY': ('apply-retention-policy', 'Continue to apply the retention window to automatically-created snapshots when the source disk is deleted.')}, default=None, help_str='Retention behavior of automatic snapshots in the event of source disk deletion.')